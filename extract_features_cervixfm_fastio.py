"""CervixFM (DINOv2 ViT-Large) coarse feature extraction — fast IO variant.

Drop-in alternative to extract_features_cervixfm.py for HDD-backed WSI roots.
Original uses a DataLoader with N workers; each worker opens its own JPEGs in
parallel, which on a single spinning disk degenerates to ~24MB/s seek-thrash
(the same pathology the RT-DETR extractor hits). This version uses the same
recipe as Detection_Distill/rtdetr/extract_features_rtdetr.py:
  - ONE reader thread streams JPEG bytes in CSV/glob order (keeps disk sequential)
  - N decoder threads PIL-decode + transform in parallel (libjpeg/PIL release GIL)
  - Main thread batches CHW tensors to the GPU
Disk-bound floor on this hardware: ~120 MB/s ≈ 200 patch/s.

Args, CSV format, output layout, idempotent skip, and model are identical to
the upstream extract_features_cervixfm.py.
"""
import argparse
import csv
import glob
import os
import queue
import sys
import threading
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.environ.setdefault("XFORMERS_DISABLED", "1")

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from model.vision_transformer import vit_large

_REDUCE_FLAG = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
                4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}
_IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def build_cervixfm(weight_path, config_path):
    cfg = OmegaConf.load(config_path)
    s = cfg.student
    model = vit_large(
        patch_size=s.patch_size,
        init_values=s.layerscale,
        ffn_layer=s.ffn_layer,
        block_chunks=s.block_chunks,
        qkv_bias=s.qkv_bias,
        proj_bias=s.proj_bias,
        ffn_bias=s.ffn_bias,
        num_register_tokens=s.num_register_tokens,
        interpolate_offset=s.interpolate_offset,
        interpolate_antialias=s.interpolate_antialias,
    )
    ckpt = torch.load(weight_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and "cls_token" not in ckpt:
        ckpt = ckpt["model"]
    info = model.load_state_dict(ckpt, strict=False)
    print(f"CervixFM weight load: missing={len(info.missing_keys)} "
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
    ap = argparse.ArgumentParser("CervixFM fast-IO extractor (1 reader + N decoders)")
    ap.add_argument("--wsi_csv", required=True,
                    help="CSV with wsi_path (and optional wsi_name) column")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--feat_dir", default="cervixfm-coarse")
    ap.add_argument("--weight_path", default=os.path.join(REPO, "pretrained", "CervixFM.pth"))
    ap.add_argument("--config_path", default=os.path.join(REPO, "config", "custom_test.yaml"))
    ap.add_argument("--inference_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=12,
                    help="decode threads (NOT readers)")
    ap.add_argument("--reduce", type=int, default=4, choices=[1, 2, 4, 8],
                    help="libjpeg reduced-color decode factor; final resize is bicubic to 224")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--amp_dtype", default="float16", choices=["float16", "bfloat16"])
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
        items.extend((wi, p) for p in files)
    total_patches = len(items)
    print(f"total patches to encode: {total_patches}", flush=True)

    print("=" * 60, flush=True)
    print(f"loading CervixFM: {args.weight_path}", flush=True)
    model = build_cervixfm(args.weight_path, args.config_path).to(device)
    print("model ready | out dim 1024 | AMP:",
          "off" if args.no_amp else args.amp_dtype, flush=True)
    print("=" * 60, flush=True)

    flag = _REDUCE_FLAG[args.reduce]
    n_dec = max(1, args.num_workers)
    raw_q = queue.Queue(maxsize=256)
    dec_q = queue.Queue(maxsize=256)

    def reader():
        for wi, p in items:
            try:
                with open(p, "rb") as f:
                    raw_q.put((wi, f.read()))
            except Exception:
                raw_q.put((wi, None))
        for _ in range(n_dec):
            raw_q.put(None)

    def decoder():
        while True:
            it = raw_q.get()
            if it is None:
                dec_q.put(None)
                return
            wi, b = it
            t = None
            if b:
                try:
                    im = cv2.imdecode(np.frombuffer(b, np.uint8), flag)
                    if im is not None:
                        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        im = (im - _IMNET_MEAN) / _IMNET_STD
                        t = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))
                except Exception:
                    t = None
            if t is None:
                t = torch.zeros(3, 224, 224)
            dec_q.put((wi, t))

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
                with torch.autocast("cuda", dtype=amp_dtype):
                    feats = model(x)
            else:
                feats = model(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        feats = feats.float().cpu().numpy()
        n_patch += len(feats)
        for f, wi in zip(feats, bw):
            buf.setdefault(wi, []).append(f)
            if len(buf[wi]) == wsi_counts[wi]:
                arr = np.stack(buf.pop(wi), axis=0)
                torch.save(torch.from_numpy(arr),
                           os.path.join(out_pt, wsi_names[wi] + ".pt"))
                done += 1
                if done % 50 == 0:
                    el = time.time() - t0
                    print(f"[r{args.local_rank}] {done}/{len(pending)} WSIs | "
                          f"{n_patch/el:.0f} patch/s ({n_patch*0.6/el:.0f}MB/s) | "
                          f"{arr.shape}", flush=True)
        bi.clear()
        bw.clear()

    finished = 0
    while finished < n_dec:
        x = dec_q.get()
        if x is None:
            finished += 1
            continue
        wi, img = x
        bi.append(img)
        bw.append(wi)
        if len(bi) >= args.inference_batch_size:
            flush()
    flush()
    print(f"[r{args.local_rank}] DONE: saved {done} WSIs, {n_patch} patches in "
          f"{(time.time()-t0)/3600:.2f}h -> {out_pt}", flush=True)


if __name__ == "__main__":
    main()
