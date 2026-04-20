# FashionCLIP 集成 — 工作内容总结

## 概述

在原有 CLIP 检索 pipeline 基础上，接入了 **FashionCLIP**（`patrickjohncyh/fashion-clip`，面向时尚领域微调的 CLIP 模型），并在 Fashion-IQ 的三个类别（dress、shirt、toptee）上跑通了完整的 Recall@K 评测。

---

## 代码改动

### 1. `extract_features.py`
新增 `encode_images_fashionclip()`，使用 FashionCLIP 提取图像 embedding：
- 绕过原库损坏的 `FashionCLIP.encode_images()` API
- 直接调用 `model.model.vision_model`，取 `pooler_output`（768 维）
- 再经 `model.model.visual_projection` 投影到 CLIP 共享空间（512 维）
- 强制使用 CPU（FashionCLIP 在 Apple Silicon MPS 上卡死）

### 2. `evaluate.py`
- 新增 FashionCLIP 文本编码路径：`text_model` + `text_projection`（同样投影到 512 维共享空间）
- **修复关键 bug**：gallery 只保留 val split 的 ASIN（约 3600–6200 张），而非全部下载的图片（约 18000+ 张）。该 bug 导致分母过大，R@K 被人为压低约 3 倍
- 新增 **R@50** 指标，所有模式和输出表格均包含
- 结果文件名带 model 标签：`eval_results_{category}_{model}.json`

### 3. FashionCLIP 库补丁
- `use_auth_token` → `token`（新版 `transformers` 已移除旧参数）
- 强制 device 为 CPU，规避 MPS 卡死

---

## 数据与特征文件

下载了三个类别的图片，提取并提交了预计算 embedding：

| 类别 | Gallery 大小 | CLIP .npy | FashionCLIP .npy |
|------|------------|-----------|-----------------|
| dress | ~3,653 | 18 MB | 35 MB |
| shirt | ~6,182 | 30 MB | 60 MB |
| toptee | ~5,261 | 26 MB | 51 MB |

所有 `.npy` 文件已提交到 `features/`，队友可直接运行 `evaluate.py`，无需重新提取。

---

## 评测结果（fusion α=0.7）

| 方法 | Dress R@10 | Dress R@50 | Shirt R@10 | Shirt R@50 | Toptee R@10 | Toptee R@50 |
|------|-----------|-----------|-----------|-----------|------------|------------|
| CLIP text-only | 12.14% | 29.34% | 16.59% | 30.19% | 18.43% | 35.62% |
| CLIP image-only | 4.03% | 11.23% | 6.96% | 14.68% | 6.70% | 14.09% |
| CLIP fusion α=0.7 | 16.39% | 34.93% | 16.85% | 31.32% | 21.05% | 37.71% |
| FashionCLIP text-only | 21.44% | 40.30% | 21.84% | 37.87% | 28.17% | 47.88% |
| FashionCLIP image-only | 5.70% | 14.40% | 9.22% | 18.70% | 7.61% | 17.78% |
| FashionCLIP fusion α=0.7 | **26.97%** | **46.64%** | **27.92%** | **45.29%** | **33.00%** | **54.37%** |

FashionCLIP 在三个类别上均明显优于 CLIP，toptee 差距最大（R@10 +12 pp）。α=0.7（偏文字权重）的 fusion 在所有情况下均优于纯文本，说明候选图像的视觉信息即使权重较低也有贡献。

---

## 复现方法

```bash
# 特征提取（如已有 .npy 文件可跳过）
python extract_features.py --model clip --category dress
python extract_features.py --model fashionclip --category dress

# 评测
python evaluate.py --model clip --category dress --alphas 0.7 --save
python evaluate.py --model fashionclip --category dress --alphas 0.7 --save
```
