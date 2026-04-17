# 项目整体架构

## 数据流向图

```
Fashion-IQ 数据集（图片 + caption 标注）
         │
         ▼
   download_images.py          ← 下载图片到 images/dress/
         │
         ▼
   extract_features.py         ← CLIP 提取图像特征（512维向量）
         │
         ▼
   features/
   ├── dress_embeddings.npy    ← shape: (N, 512)，每行一张图的向量
   └── dress_paths.txt         ← 对应图片路径，行号和 embeddings 行号一一对应
         │
         ▼
   build_index.py              ← 把向量塞进 FAISS 索引，加速检索
         │
         ▼
   features/dress_index.faiss
```

之后三条检索线都依赖上面这个 pipeline：

```
text_retrieval.py    → 文字 → CLIP text encoder → 检索 gallery
fusion_retrieval.py  → 文字 + 图片 → 融合分数 → 检索 gallery
evaluate.py          → 批量跑 Recall@K，对比三种模式
compare_viz.py       → 生成报告用的对比图
```

## 完整运行顺序

```bash
# Step 1（Member A 的工作，你不用做，但要确认 images/ 存在）
python download_images.py
python extract_features.py
python build_index.py

# Step 2 — Member B
python text_retrieval.py                        # freeform + FashionIQ demo
python query_analysis.py                        # 生成 query_analysis_grid.png

# Step 3 — Member C
python evaluate.py --save                       # 核心评估，输出 Recall@K 表格
python fusion_retrieval.py --compare --n 5      # 生成 demo 对比图
python compare_viz.py --scan 300                # 生成成功/失败案例图
```

## 参数需要手动调吗？

**不需要。** 所有关键参数都有合理默认值，而且调参过程已内置在脚本里。

| 脚本 | 会自动测试的参数 | 你需要做的 |
|------|----------------|-----------|
| `evaluate.py` | `--alphas 0.3 0.5 0.7` 三个都跑 | 看输出表格，找 R@10 最高的 alpha |
| `fusion_retrieval.py` | 默认 `alpha=0.5` | 把上一步找到的最好 alpha 传进来 |
| `compare_viz.py` | 默认 `alpha=0.5` | 同上 |
| `text_retrieval.py` | 无调参 | 直接跑就行 |

举例：如果 `evaluate.py` 输出显示 `fusion α=0.5` 最好，那就：
```bash
python compare_viz.py --alpha 0.5 --scan 300
python fusion_retrieval.py --compare --alpha 0.5 --n 5
```

## 模块分工

| 文件 | 成员 | 核心功能 |
|------|------|---------|
| `download_images.py` | A | 下载图片 |
| `extract_features.py` | A | CLIP 图像特征提取 |
| `build_index.py` | A | 建 FAISS 向量索引 |
| `text_retrieval.py` | B | 文本→图像检索 |
| `query_analysis.py` | B | 查询类型对比分析图 |
| `evaluate.py` | C | Recall@K 批量评估 |
| `fusion_retrieval.py` | C | 多模态融合检索 demo |
| `compare_viz.py` | C | 报告质量对比可视化 |

## 技术依赖

```
openai-clip    → CLIP 模型（ViT-B/32）
faiss-cpu      → 向量检索索引
torch          → 深度学习推理
numpy          → 矩阵运算
matplotlib     → 结果可视化
Pillow         → 图片读取
tqdm           → 进度条
```
