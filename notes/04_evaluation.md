# 评估指标：Recall@K

> 相关代码：`evaluate.py`

## Recall@K 是什么

对每一条 query，检索后看：**target 图片有没有出现在前 K 个结果里？**

```
Recall@1  = target 是 top-1 结果 → 1 分，否则 0 分
Recall@5  = target 在 top-5 里   → 1 分，否则 0 分
Recall@10 = target 在 top-10 里  → 1 分，否则 0 分

最终 Recall@K = 所有 query 的平均分（0.0 ~ 1.0）
```

## 为什么不用 Accuracy / Top-1 精度

每条 query 的 gallery 有 ~4000 张图，随机猜中率只有 0.025%。  
如果只看 top-1，数字会很低（约 5-8%），不好体现模型的实际能力。

Recall@K 的优点：
- 体现"在 K 次机会里至少找到一次"的能力
- K=5 和 K=10 能看出模型的 recall 提升速度
- 是 Fashion-IQ 论文和 leaderboard 的标准指标

## evaluate.py 里的快速计算

`evaluate.py` 用矩阵运算一次算出所有 query 的 rank，避免逐条循环：

```python
# text_scores: shape (N, G)   — N 条 query，G 张 gallery 图
# target_idx:  shape (N,)     — 每条 query 对应的目标图片在 gallery 里的索引

target_scores = text_scores[np.arange(N), target_idx]   # (N,) 每条 query 中 target 的分数

ranks = (text_scores > target_scores[:, None]).sum(axis=1) + 1
```

**最后一行分解来看：**

```python
target_scores[:, None]              # (N,) → (N, 1)，广播用
text_scores > target_scores[:, None]  # (N, G) 布尔矩阵，True = 某图比 target 分数高
.sum(axis=1)                        # (N,) 对每条 query 数有几张图比 target 分数高
+ 1                                 # 变成 1-indexed rank（0 张图比它高 → rank=1 → 最好）
```

**举例：**
- 如果 target 是 gallery 里分数最高的那张 → ranks[i] = 1（最好）
- 如果有 50 张图分数比 target 高 → ranks[i] = 51

然后 Recall@K 就是：
```python
recall_at_k = (ranks <= K).mean()   # 有几条 query 的 rank ≤ K
```

## 为什么这么快

核心是这两行 matmul：

```python
text_scores  = text_embs  @ gallery_emb.T   # (N, G)
image_scores = image_embs @ gallery_emb.T   # (N, G)
```

以 dress val 为例：N=~1800，G=~4000，D=512

```
(1800, 512) @ (512, 4000) = (1800, 4000)
```

这是一个 ~3.7M 个浮点运算的矩阵乘法，NumPy 在 CPU 上 **< 1 秒** 完成。  
所以 evaluate.py 跑完整个 val set 通常只需要几分钟（主要耗时在 CLIP encode_text）。

## 期望结果范围

CLIP zero-shot（没有任何 fine-tuning）在 Fashion-IQ dress val 上的参考值：

| 模式 | R@1 | R@5 | R@10 |
|------|-----|-----|------|
| text-only | 5–8% | 16–22% | 25–33% |
| image-only | 4–6% | 13–18% | 20–28% |
| fusion α=0.5 | 7–11% | 20–27% | 30–40% |

数字不高，因为我们完全没有 fine-tuning。  
但关键是：**fusion > 单模态**，这就是论文里要说的主要结论。

## 如何找最好的 alpha

```bash
python evaluate.py --alphas 0.3 0.5 0.7 --save
```

输出表格类似：

```
  Mode                    R@1      R@5     R@10       N
  text-only             6.84%   19.42%   28.91%    1823
  image-only            5.21%   15.78%   24.33%    1823
  fusion α=0.3          7.53%   21.89%   32.45%    1823
  fusion α=0.5          8.12%   23.11%   34.87%    1823   ← 最好
  fusion α=0.7          7.91%   22.34%   33.20%    1823
```

选 R@10 最高的 alpha（通常是 0.5），然后用它生成报告图。
