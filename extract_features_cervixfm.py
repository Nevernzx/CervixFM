"""
CervixFM (DINOv2 ViT-Large) 特征提取 - coarse 模式
每个 WSI 一个 patch 目录(内含 x_y.jpg 大图块)，每张大图块 -> resize 224 -> 一个 1024-d 特征。
与 uni2-coarse / gigapath-coarse / unicas-coarse 对齐(一块=一特征, 无裁剪, 无过滤)。

用法:
  CUDA_VISIBLE_DEVICES=1 <new-mil python> extract_features_cervixfm.py \
      --wsi_csv /path/gc20k6_rand50.csv \
      --output_path /data/.../gc20k6_rand50_features --feat_dir cervixfm-coarse
"""
import os
import sys
import glob
import csv
import time
import argparse

# 仓库根目录 + 使用 pure-torch 回退(无需 xformers)
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.environ.setdefault("XFORMERS_DISABLED", "1")

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

from model.vision_transformer import vit_large

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
          f"unexpected={len(info.unexpected_keys)}")
    if info.missing_keys:
        print("  missing[:5]:", info.missing_keys[:5])
    if info.unexpected_keys:
        print("  unexpected[:5]:", info.unexpected_keys[:5])
    return model.eval()


class CoarseTileDataset(Dataset):
    """一个 WSI 目录: 每个 x_y.jpg 大图块 -> resize224 -> 一个 patch (coarse)。"""
    def __init__(self, wsi_path, transform):
        self.transform = transform
        files = glob.glob(os.path.join(wsi_path, "*.jpg")) + \
                glob.glob(os.path.join(wsi_path, "*.png"))
        try:
            files = sorted(files, key=lambda x: (
                int(os.path.basename(x).split(".")[0].split("_")[0]),
                int(os.path.basename(x).split(".")[0].split("_")[1])))
        except Exception:
            files = sorted(files)
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            t = self.transform(img)
            img.close()
            return t
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return None


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.stack(batch, dim=0)


@torch.no_grad()
def extract_one(wsi_dir, model, transform, args):
    ds = CoarseTileDataset(wsi_dir, transform)
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=args.inference_batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=True, drop_last=False,
                        persistent_workers=False)
    feats = []
    use_amp = not args.no_amp and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    for batch in loader:
        if batch is None:
            continue
        batch = batch.to(device, non_blocking=True)
        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                out = model(batch)
        else:
            out = model(batch)
        if isinstance(out, (tuple, list)):
            out = out[0]
        feats.append(out.float().cpu())
    if not feats:
        return None
    return torch.cat(feats, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi_csv", required=True, help="含 wsi_path 列的 csv (每个 WSI 的 patch 目录)")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--feat_dir", default="cervixfm-coarse")
    ap.add_argument("--weight_path", default=os.path.join(REPO, "pretrained", "CervixFM.pth"))
    ap.add_argument("--config_path", default=os.path.join(REPO, "config", "custom_test.yaml"))
    ap.add_argument("--inference_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--amp_dtype", default="float16", choices=["float16", "bfloat16"])
    args = ap.parse_args()

    out_pt = os.path.join(args.output_path, args.feat_dir, "pt")
    os.makedirs(out_pt, exist_ok=True)
    done = set(os.listdir(out_pt))

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    print("=" * 60)
    print(f"加载 CervixFM: {args.weight_path}")
    model = build_cervixfm(args.weight_path, args.config_path).to(device)
    print("模型加载成功 | 输出维度 1024 | AMP:",
          "off" if args.no_amp else args.amp_dtype)
    print("=" * 60)

    rows = list(csv.DictReader(open(args.wsi_csv)))
    total = len(rows)
    ok = skip = fail = 0
    for i, r in enumerate(rows):
        wsi_dir = r["wsi_path"]
        name = r.get("wsi_name") or os.path.basename(wsi_dir.rstrip("/"))
        if name + ".pt" in done:
            skip += 1
            print(f"[{i+1}/{total}] 跳过已处理: {name}")
            continue
        t0 = time.time()
        feat = extract_one(wsi_dir, model, transform, args)
        if feat is None:
            fail += 1
            print(f"[{i+1}/{total}] 失败(无patch): {name}")
            continue
        torch.save(feat, os.path.join(out_pt, name + ".pt"))
        ok += 1
        print(f"[{i+1}/{total}] {name}: {tuple(feat.shape)} | {time.time()-t0:.1f}s")

    print("=" * 60)
    print(f"完成! 成功 {ok} | 跳过 {skip} | 失败 {fail}")
    print("=" * 60)


if __name__ == "__main__":
    main()
