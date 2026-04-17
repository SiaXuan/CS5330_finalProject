# 概念：shape（张量的形状）

shape 就是"这块数据有几层、每层几个"，用一个括号里的数字列表表示。

## 从一维开始

```python
[1, 2, 3, 4, 5, 6, 7]
# shape = (7,)   ← 一维，7 个元素
```

## 二维：表格 / 矩阵

```python
[[1, 2, 3, 4],
 [5, 6, 7, 8],
 [9, 0, 1, 2]]
# shape = (3, 4)  ← 3 行 4 列
```

读法：**括号里从左到右，依次是"最外层有几个"、"次外层有几个"……**

## 本项目里遇到的 shape

| shape | 含义 |
|-------|------|
| `(512,)` | 512 个数，一个向量 |
| `(1, 512)` | 1 行 512 列，一张图片的向量（包了 batch 层） |
| `(N, 512)` | N 行 512 列，N 张图片各自的向量 → gallery embeddings |
| `(N, G)` | N 条 query × G 张 gallery 图片的相似度分数矩阵 |

## 为什么 (1, 512) 不是 (512,)

CLIP 处理单张图片时，输出其实是 `(1, 512)`，因为 PyTorch 里的模型习惯批量处理，
哪怕只有一张图，也要包一层"batch 维"表示"这是 1 个样本的 batch"。

`unsqueeze(0)` 就是手动加上这一层：`(512,) → (1, 512)`。

→ 详见 [concepts/batch.md](batch.md)

## 怎么查某个变量的 shape

```python
import numpy as np
import torch

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)   # (2, 3)

t = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t.shape)   # torch.Size([2, 3])
```
