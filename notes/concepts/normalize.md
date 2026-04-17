# 概念：L2 归一化（normalize）

## 数学含义

把一个向量除以它自己的模（长度），让它变成一个长度为 1 的单位向量：

```
v_normalized = v / ||v||

其中 ||v|| = √(v₁² + v₂² + ... + v₅₁₂²)
```

归一化前后方向不变，只是长度统一变成 1。

## 为什么要归一化

归一化之后，两个向量的**内积（dot product）= cosine 相似度**：

```
a · b = |a||b| cos(θ)

如果 |a| = |b| = 1，那么 a · b = cos(θ)
```

cosine 相似度只看两个向量的**方向**是否一致，不受向量长度影响，
值域固定在 [-1, 1]，便于比较。

## PyTorch 写法拆解

```python
emb = emb / emb.norm(dim=-1, keepdim=True)
```

**`emb.norm(dim=-1, keepdim=True)`** — 分三步理解：

| 部分 | 含义 |
|------|------|
| `.norm(...)` | 计算 L2 范数（模），即 √(x₁² + x₂² + ... + xₙ²) |
| `dim=-1` | 对**最后一个维度**算，这里就是对那 512 个数算模 |
| `keepdim=True` | 保留维度，让结果 shape 从 `(1,)` 变成 `(1, 1)`，这样才能广播着除 `(1, 512)` |

如果 `emb` shape 是 `(1, 512)`：
```
emb.norm(dim=-1, keepdim=False) → shape (1,)    ← 没法直接除 (1, 512)
emb.norm(dim=-1, keepdim=True)  → shape (1, 1)  ← 可以广播，每列都除这个数 ✓
```

## NumPy 里的等价写法

```python
# NumPy（evaluate.py 里用的）
emb = emb / np.linalg.norm(emb)          # 单个向量
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # 矩阵，每行归一化
```

## 归一化之后能直接用内积当相似度

```python
# 两个已归一化的向量
score = np.dot(a, b)          # 值域 [-1, 1]，越大越相似
# 或者矩阵形式（gallery 检索）
scores = gallery_emb @ query_emb   # (N,)，每张图的相似度
```
