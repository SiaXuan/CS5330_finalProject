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
| `.norm(...)` | 只算模，不做除法。即 √(x₁² + x₂² + ... + xₙ²) |
| `dim=-1` | 对最后一个维度算，把那一维的所有数折叠成一个模值 |
| `keepdim=True` | 保留被折叠的那个维度（变成 size 1），用于广播 |

**`dim=-1` 的直觉：**

```
shape (2, 3)，dim=-1 就是对每行的 3 个数各算一个模：

[[1, 2, 3],     norm(dim=-1)      [[3.74],
 [4, 5, 6]]    ──────────────→     [8.77]]

(2, 3)                              (2, 1)  ← keepdim=True
```

每行 3 个数 → 每行 1 个模，行数不变。

**`keepdim` 的作用（广播）：**

广播 = 形状不同时自动对齐，让小的那个扩展去匹配大的。

```
emb       shape (1, 512)   → 1行512个数
magnitude shape (1,  1 )   → 1行1个数

除的时候，(1,1) 自动扩展成 (1, 512)，每个数都除同一个模值：
[[m, m, m, ..., m]]  ← 512 个相同的 m
```

如果 `keepdim=False`，magnitude shape 变成 `(1,)`，PyTorch 不知道怎么对齐 `(1, 512)` 和 `(1,)`，结果不对。

本项目实际的 `emb` shape 是 `(1, 512)`：
```
emb.norm(dim=-1, keepdim=False) → shape (1,)    ← 广播对不齐
emb.norm(dim=-1, keepdim=True)  → shape (1, 1)  ← 能广播 ✓
```

## NumPy 里的等价写法

`np.linalg` 是 NumPy 的线性代数（**lin**ear **alg**ebra）子模块，
专门放矩阵运算相关的函数，`.norm()` 就是其中算模的那个。

**单个向量：**

```python
v = np.array([3.0, 4.0])
np.linalg.norm(v)        # √(3² + 4²) = 5.0

v / np.linalg.norm(v)    # [0.6, 0.8]  ← 长度变成 1
```

**矩阵，每行归一化（evaluate.py 里的用法）：**

```python
emb = np.array([[3.0, 4.0],   # 行0，模=5
                [1.0, 0.0]])  # 行1，模=1

np.linalg.norm(emb, axis=1, keepdims=True)
# → [[5.0],   ← 行0的模
#    [1.0]]   ← 行1的模
# shape (2, 1)，keepdims 和 PyTorch 的 keepdim 是一个意思

emb / np.linalg.norm(emb, axis=1, keepdims=True)
# → [[0.6, 0.8],   ← 行0除以5
#    [1.0, 0.0]]   ← 行1除以1
```

`axis=1` 等价于 PyTorch 的 `dim=-1`，都是"对每行操作"。

## 归一化之后能直接用内积当相似度

```python
# 两个已归一化的向量
score = np.dot(a, b)          # 值域 [-1, 1]，越大越相似
# 或者矩阵形式（gallery 检索）
scores = gallery_emb @ query_emb   # (N,)，每张图的相似度
```
