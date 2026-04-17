# 融合策略与调参

> 相关代码：`fusion_retrieval.py`, `evaluate.py`, `compare_viz.py`

## Fusion 公式

```python
final_score = alpha * text_score + (1 - alpha) * image_score
```

- `alpha = 1.0` → 纯文本检索
- `alpha = 0.0` → 纯图像检索
- `alpha = 0.5` → 各占一半（最常用的起点）

text_score 和 image_score 都是 cosine 相似度（归一化后的内积），值域相同，可以直接加权平均。

## 为什么要 Fusion

文本和图像各有优劣：

| 模态 | 擅长捕捉 | 容易被迷惑 |
|------|---------|-----------|
| 文本 | 语义变化："无袖"、"纯色"、"更长" | 视觉相似但描述模糊的图片 |
| 图像 | 整体风格、版型、材质、颜色 | 和 candidate 视觉很像的其他图 |

两者互补：文本不确定的地方，图像来补；图像区分不开的时候，文本来拯救。  
→ Fusion 后 Recall@K 通常比单模态高 3-8 个百分点。

## 调参已内置，不需要手动试

```bash
python evaluate.py --alphas 0.3 0.5 0.7
```

一次运行就输出三种 alpha 的 Recall@K 表格，对比着看哪个最好。  
不需要自己改代码。

## 调完参之后的操作

假设 `evaluate.py` 结果显示 `alpha=0.5` 最好：

```bash
# 生成报告对比图（3行：text-only / image-only / fusion）
python fusion_retrieval.py --compare --alpha 0.5 --n 5

# 生成成功/失败案例网格
python compare_viz.py --alpha 0.5 --scan 300
```

## compare_viz.py 自动找三类案例

| 文件 | 案例类型 | 报告用途 |
|------|---------|---------|
| `success_cases.png` | fusion rank 明显低于 text_rank 和 image_rank | 展示 fusion 的优势 |
| `failure_cases.png` | 三种模式 rank 都 > 20 | 分析失败原因 |
| `baseline_wins.png` | text 或 image rank ≤ 5，但 fusion 更差 | 分析 fusion 什么时候 hurt |

## Fusion 什么时候会 hurt（failure analysis）

**案例 1：Caption 太模糊**
- Caption 是 "is longer"、"is more casual" 这类模糊描述
- 文本向量噪声大，拉低了 fusion 分数
- → 图像单独跑更好

**案例 2：Candidate 和 target 视觉太像**
- 两者颜色、款式非常接近，只有细节不同
- 图像检索把 candidate 附近的图排得很高（大量视觉相似的图）
- Target 被淹没在相似图堆里
- → 文本单独跑更好

**案例 3：Caption 描述的是 target 没有的特征**
- 标注人描述 candidate 的特征，而不是 target 的
- 这是数据集标注质量问题，模型无法应对

这些分析写进报告里就是 "Limitations" 和 "Failure Analysis" 部分。

## 可以尝试的其他 alpha 值

`--alphas 0.1 0.3 0.5 0.7 0.9` 更细粒度地扫描，但本项目报告里 0.3/0.5/0.7 三个点就够了。
