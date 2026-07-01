"""UniCAS (timm ViT-L/16, DINOv2/SwiGLU) coarse feature extraction — fast IO, PIL pipeline.

Preprocessing matches the lab's extract_features_FM.py (PIL decode + torchvision
Resize(224, BICUBIC, antialiased) + ToTensor + ImageNet norm), so the resulting
unicas-coarse features are consistent with the lab's other coarse feature sets
(per-patch cosine vs the reference gc20k6_rand50 unicas-coarse: 0.9977 with PIL `draft`
fast decode, 0.9989 with full decode).

Keeps the HDD-friendly architecture: ONE sequential byte reader + N PIL-decode threads,
with within-WSI position preserved so rows stay in sorted patch order.
Two bugs an earlier cv2 fastio variant had (both fixed here): (1) accumulating features
in decode-completion order instead of sorted order (dominant, cos-limited to ~0.88),
(2) cv2.resize does not antialias on downscale (cos ~0.82). PIL `draft` gives ~10x
faster decode than full PIL decode while staying antialiased.

Output: <output>/<feat_dir>/pt/<wsi>.pt  shape (N_tiles, 1024) fp32.

USAGE (other server) -----------------------------------------------------------
Deps:    torch, timm>=1.0 (has timm.layers.mlp.GluMlp), torchvision, Pillow.
         (validated with timm 1.0.15 / torch 1.11; env "tct-info" here.)
Weights: UniCAS.pth (~1.2 GB) from HuggingFace `jianght/UniCAS`. NOT in git.
         Put it at  <this_dir>/pretrained/UniCAS/UniCAS.pth  (default), or pass --weight_path.
CSV:     --wsi_csv points to a CSV with a `wsi_path` column (and optional `wsi_name`),
         each row = one WSI's patch directory containing x_y.jpg tiles (same layout as
         the CervixFM extractor / gc20k-6.local.csv).
Run:
    CUDA_VISIBLE_DEVICES=0 python extract_features_unicas_fastio.py \
        --wsi_csv <your.csv> --output_path <out> --feat_dir unicas-coarse \
        --num_workers 12 --inference_batch_size 256
Keep --inference_batch_size 256 / --num_workers 12 / bf16 to stay consistent.
Idempotent: already-written .pt are skipped, so it is resumable.
"""
import argparse
import csv
import functools
import glob
import io
import os
import queue
import threading
import time

import torch
import timm
from PIL import Image
from torchvision import transforms
from timm.layers.mlp import GluMlp

_DEFAULT_WEIGHT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "pretrained", "UniCAS", "UniCAS.pth")

# Lab preprocessing (extract_features_FM.py UnicasBackbone.preprocess_val): PIL resize
# to 224 is antialiased by PIL for BICUBIC on downscale.
_TF = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def build_unicas(weight_path):
    params = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, init_values=1e-05,
        mlp_ratio=2.671875 * 2, mlp_layer=functools.partial(GluMlp, gate_last=False),
        act_layer=torch.nn.SiLU, no_embed_class=False, img_size=224,
        num_classes=0, in_chans=3,
    )
    model = timm.models.VisionTransformer(**params)
    st = torch.load(weight_path, map_location="cpu")
    if isinstance(st, dict) and "model" in st and "cls_token" not in st:
        st = st["model"]
    info = model.load_state_dict(st, strict=False)
    print(f"UniCAS weight load: missing={len(info.missing_keys)} "
          f"unexpected={len(info.unexpected_keys)}", flush=True)
    return model.eval()


def _sorted_patches(wsi_dir):
    files = (glob.glob(os.path.join(wsi_dir, "*.jpg"))
             + glob.glob(os.path.join(wsi_dir, "*.png")))
    try:
        return sorted(files, key=lambda p: (
            int(os.path.basename(p).split(".")[0].split("_")[0]),
            int(os.path.basename(p).split(".")[0].split("_")[1])))
    except Exception:
        return sorted(files)


def main():
    ap = argparse.ArgumentParser("UniCAS fast-IO extractor (PIL pipeline, 1 reader + N decoders)")
    ap.add_argument("--wsi_csv", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--feat_dir", default="unicas-coarse")
    ap.add_argument("--weight_path", default=_DEFAULT_WEIGHT)
    ap.add_argument("--inference_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=12, help="PIL decode threads")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--amp_dtype", default="bfloat16", choices=["float16", "bfloat16"],
                    help="lab default is bfloat16")
    ap.add_argument("--local_rank", type=int, default=0)
    args = ap.parse_args()

    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")
    torch.backends.cudnn.benchmark = True

    out_pt = os.path.join(args.output_path, args.feat_dir, "pt")
    os.makedirs(out_pt, exist_ok=True)
    done_set = set(os.listdir(out_pt))

    rows = list(csv.DictReader(open(args.wsi_csv)))
    pending = []
    for r in rows:
        wsi_dir = r["wsi_path"]
        name = r.get("wsi_name") or os.path.basename(wsi_dir.rstrip("/"))
        if name + ".pt" in done_set:
            continue
        pending.append((name, wsi_dir))
    print(f"total WSIs: {len(rows)} | pending: {len(pending)} | out: {out_pt}", flush=True)
    if not pending:
        print("nothing to do", flush=True)
        return

    items, wsi_names, wsi_counts = [], [], []
    for wi, (name, d) in enumerate(pending):
        files = _sorted_patches(d)
        wsi_names.append(name)
        wsi_counts.append(len(files))
        # carry within-WSI position pj so features are saved in sorted patch order,
        # not decode-completion order (parallel decoders finish out of order).
        items.extend((wi, pj, p) for pj, p in enumerate(files))
    total_patches = len(items)
    print(f"total patches to encode: {total_patches}", flush=True)

    print("=" * 60, flush=True)
    print(f"loading UniCAS: {args.weight_path}", flush=True)
    model = build_unicas(args.weight_path).to(device)
    print("model ready | out dim 1024 | pipeline PIL+antialias | AMP:",
          "off" if args.no_amp else args.amp_dtype, flush=True)
    print("=" * 60, flush=True)

    n_dec = max(1, args.num_workers)
    raw_q = queue.Queue(maxsize=256)
    dec_q = queue.Queue(maxsize=256)

    def reader():
        for wi, pj, p in items:
            try:
                with open(p, "rb") as f:
                    raw_q.put((wi, pj, f.read()))
            except Exception:
                raw_q.put((wi, pj, None))
        for _ in range(n_dec):
            raw_q.put(None)

    def decoder():
        while True:
            it = raw_q.get()
            if it is None:
                dec_q.put(None)
                return
            wi, pj, b = it
            t = None
            if b:
                try:
                    im = Image.open(io.BytesIO(b))
                    im.draft("RGB", (224, 224))  # libjpeg reduced-scale decode: ~10x faster,
                    t = _TF(im.convert("RGB"))    # cos 0.9977 vs full-decode lab pipeline
                except Exception:
                    t = None
            if t is None:
                t = torch.zeros(3, 224, 224)
            dec_q.put((wi, pj, t))

    threading.Thread(target=reader, daemon=True).start()
    for _ in range(n_dec):
        threading.Thread(target=decoder, daemon=True).start()

    use_amp = (not args.no_amp) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    buf, done, n_patch, t0 = {}, 0, 0, time.time()
    bi, bw = [], []

    def flush():
        nonlocal done, n_patch
        if not bi:
            return
        x = torch.stack(bi).to(device, non_blocking=True)
        with torch.inference_mode():
            if use_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    feats = model(x)
            else:
                feats = model(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        feats = feats.float().cpu()
        n_patch += feats.shape[0]
        for k in range(feats.shape[0]):
            wi, pj = bw[k]
            buf.setdefault(wi, {})[pj] = feats[k]
            if len(buf[wi]) == wsi_counts[wi]:
                d = buf.pop(wi)
                arr = torch.stack([d[j] for j in range(wsi_counts[wi])], dim=0)
                torch.save(arr, os.path.join(out_pt, wsi_names[wi] + ".pt"))
                done += 1
                if done % 50 == 0:
                    el = time.time() - t0
                    print(f"[r{args.local_rank}] {done}/{len(pending)} WSIs | "
                          f"{n_patch/el:.0f} patch/s | {tuple(arr.shape)}", flush=True)
        bi.clear()
        bw.clear()

    finished = 0
    while finished < n_dec:
        x = dec_q.get()
        if x is None:
            finished += 1
            continue
        wi, pj, img = x
        bi.append(img)
        bw.append((wi, pj))
        if len(bi) >= args.inference_batch_size:
            flush()
    flush()
    print(f"[r{args.local_rank}] DONE: saved {done} WSIs, {n_patch} patches in "
          f"{(time.time()-t0)/3600:.2f}h -> {out_pt}", flush=True)


if __name__ == "__main__":
    main()
