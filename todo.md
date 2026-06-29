# TODO — 给 server A 的 Claude（CervixFM coarse 特征抽取）

> 这是双服务器分工的第二个任务。本文件由 server B（=3080_01，持有 volume1/2/3/dep13）
> 上的 Claude 写给 server A（持有 volume4/5/dep45）。你的工作：用与 server B **完全一致**
> 的 CervixFM 配置，在 server A 本地把 `gc20k-6.csv` 在你这边的子集特征抽出来。
> 第一版 RT-DETR 的 todo 是 git commit `f2449b4`，已完成。

## 背景（server B 已完成 / 接近完成）

- CervixFM = ViT-Large DINOv2，**coarse 模式**：每张 WSI 大图块 → 224×224 → 1024-d 特征（一块一向量），与现有 `gigapath-coarse / uni2-coarse` 对齐。
- 数据集 gc20k 共 19972 WSI（CSV：`MIL_BASELINE/datasets/gc20k-6.csv`）。
- server B 本地匹配的 13336 WSI 已抽完，产物：
  `/home/25_niezhengxin/wsi-features/CervixFM-features/cervixfm-coarse/pt/<wsi_name>.pt`，shape `(N_patch, 1024)` fp32。
- 你这边大约 ~6636 WSI（CSV 总数 − server B 子集，估算）在 volume4/5/dep45 上。

## 你要做的事

### 1. 拿到 CervixFM 代码 + 权重 + fast-IO 脚本

你本机已经有 `/home/25_niezhengxin/workplace/CervixFM/`（rsync 过去的），但缺我新写的 `extract_features_cervixfm_fastio.py`。两条路：

(a) 从 server B 直接 rsync 单文件（最快）：
```
rsync -avP -e 'ssh -p <port>' \
  25_niezhengxin@<server_b_ip>:/home/25_niezhengxin/workplace/CervixFM/extract_features_cervixfm_fastio.py \
  /home/25_niezhengxin/workplace/CervixFM/
```

(b) 或从 git 拉（push 完成后）：
```
cd /home/25_niezhengxin/workplace/CervixFM
git remote set-url origin git@github.com:Nevernzx/CervixFM.git
git fetch origin && git checkout main && git pull
```

权重 `pretrained/CervixFM.pth`（1.16 GB）不在 git。本机有就直接用；没有就从 server B `scp` 过来。

### 2. 环境

需要 `torch + opencv + omegaconf`。`huawei` 环境若缺 `omegaconf`：
```
conda run -n huawei pip install omegaconf
```

### 3. 生成本机 wsi 列表 CSV

```bash
# csv-id 列表
awk -F',' 'NR>1 {n=$1; sub(".*/","",n); sub("\\.pt$","",n); print n}' \
  <D2VFM-lgj路径>/MIL_BASELINE/datasets/gc20k-6.csv > /tmp/csv_names.txt

# 本机 WSI 目录（用 find 避免空格目录名出问题）
> /tmp/local_paths.tsv
for v in volume4 volume5 volume-deprecated45; do
  d=/data/wsi/TCTGC50k/TCTGC50k-$v
  [ -d "$d" ] && find "$d" -mindepth 1 -maxdepth 1 -type d -printf '%f\t%p\n' >> /tmp/local_paths.tsv
done

# 交集 → CervixFM 脚本要的格式
echo "wsi_name,wsi_path" > /home/25_niezhengxin/workplace/CervixFM/gc20k-6.local.csv
awk -F'\t' 'NR==FNR{p[$1]=$2; next} ($1 in p){print $1","p[$1]}' \
  /tmp/local_paths.tsv /tmp/csv_names.txt >> /home/25_niezhengxin/workplace/CervixFM/gc20k-6.local.csv
wc -l /home/25_niezhengxin/workplace/CervixFM/gc20k-6.local.csv   # 应该 6000-7000 量级
```

### 4. 跑抽取（⚠️ 单进程单卡）

```
conda run -n huawei python /home/25_niezhengxin/workplace/CervixFM/extract_features_cervixfm_fastio.py \
  --wsi_csv /home/25_niezhengxin/workplace/CervixFM/gc20k-6.local.csv \
  --output_path <你的可写盘>/CervixFM-features \
  --feat_dir cervixfm-coarse \
  --num_workers 12 --inference_batch_size 256 --reduce 2
```

**两机必须一致的参数**（否则特征不可比）：
- `--reduce 2`（reduce_4 在 outlier patch 上 cos 掉到 0.96，server B 实测过；reduce_2 mean 0.9996 / min 0.989 对照 full-decode）
- `--inference_batch_size 256`
- `--num_workers 12`

产物：`<output_path>/cervixfm-coarse/pt/<wsi_name>.pt`，shape `(N, 1024)` fp32。
脚本幂等可断点续传（已存 .pt 跳过）。

## ⚠️ 关键坑（server B 踩过的）

1. **绝不用多 worker DataLoader 跑 HDD**。CervixFM 官方脚本默认 `num_workers=8` 是 SSD 场景；HDD 上 N 个 worker 各自开文件 → 寻道颠簸 → 实测 42 patch/s。`extract_features_cervixfm_fastio.py` 是 **1 顺序 reader + N decode 线程**的重写，能跑磁盘地板 ~230 patch/s。
2. **`torch.inference_mode()` 必须包**。fast-IO 脚本里已经在 `flush()` 里包好，别动；去掉显存翻倍，batch=256 直接 OOM。
3. **GPU 健康监控**。server B 跑到一半 GPU 0 fell off the bus（NVRM Xid 79），进程 CUDA 调用永久 hang 不报错（白等 4h）。建议每 30 分钟 `nvidia-smi` 看一眼；hang 了 kill 进程 + 切别的 GPU 重启（脚本幂等）。
4. **输出盘**。特征 ~5 MB/WSI × 6636 ≈ **35 GB**。`/data` 在 server B 是别人的不可写，在 server A 自己看；`/home` 通常稳。

## ETA

- 单卡 3080 ViT-L fp16 算力 ~290 patch/s（GPU 微基准）
- HDD 顺序读 ~230 patch/s（磁盘地板，瓶颈在这）
- 你这边 ~6636 WSI × ~626 patches/WSI ≈ **4.2M patches → ~5-6 小时**

## ⚠️ 已知未决问题（与第一版 RT-DETR todo 相同）

- MIL 的 split CSV 和现有 gigapath / uni2 特征都用 csv-id 命名；本次 CervixFM 产物按 **wsi 目录名** 命名，与 csv-id 只有约 2/3 直配。剩余靠用户给出「目录名 ↔ csv-id 映射表」后统一改名/重抽。这次先把直配子集抽出来。

## 完成后

把 `<output>/cervixfm-coarse/pt/` 全部回传 server B 汇总：
```
rsync -avP <output>/cervixfm-coarse/pt/ \
  25_niezhengxin@<server_b_ip>:/home/25_niezhengxin/wsi-features/CervixFM-features/cervixfm-coarse/pt/
```
（**只传特征**，不要传图像/权重）

## 一句话总结
`rsync fastio.py → pip install omegaconf → 生成 local.csv → 单进程单卡 fastio --reduce 2 → 回传特征`。两机 `reduce / inference_batch_size / num_workers` 必须一致。
