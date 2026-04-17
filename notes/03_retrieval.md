# 检索原理

> 相关代码：`extract_features.py`, `build_index.py`, `retrieve.py`, `text_retrieval.py`

## 核心思路

把所有图片都变成向量，存下来（gallery embeddings）。  
来了一个 query（文字或图片），也变成向量。  
找 gallery 里和 query 向量**最接近**的那些 → 就是检索结果。

## 流程

```
① 离线建库（跑一次，存下来）：
   所有图片 → encode_image → (N, 512) gallery_emb
   保存到 features/dress_embeddings.npy

② 在线检索（每次 query 都跑）：
   query 文字 → encode_text → (512,) query_emb
   query 图片 → encode_image → (512,) query_emb
              ↓
   scores = gallery_emb @ query_emb   # (N,) 每张图的相似度分数
              ↓
   top-k 索引 = argsort(scores)[::-1][:k]
              ↓
   返回对应图片路径
```

## 为什么用矩阵乘法

`gallery_emb` shape 是 `(N, 512)`，`query_emb` shape 是 `(512,)`：  
> → 不懂 shape？[concepts/shape.md](concepts/shape.md)

```python
scores = gallery_emb @ query_emb   # (N, 512) @ (512,) → (N,)
```

一行代码算出所有 N 张图的相似度，比 for 循环快 **100 倍以上**（NumPy/PyTorch 底层用 BLAS 加速）。  
`evaluate.py` 批量评估时甚至用 `(N_query, G)` 的矩阵，一次算完所有 query 的所有 gallery 分数。

## FAISS：IndexFlatIP

`build_index.py` 用 FAISS 把 gallery embeddings 包装成索引：

```python
index = faiss.IndexFlatIP(512)  # IP = Inner Product（内积）
index.add(embeddings)           # 把所有向量加进去
scores, indices = index.search(query_emb, top_k)  # 搜索最近的 top_k 个
```

- `IndexFlatIP` = 穷举内积（和 NumPy 矩阵乘法效果完全一样）
- embeddings 已经归一化，所以内积 = cosine 相似度
- FAISS 的优势在于支持 GPU 加速和 approximate nearest neighbor，小数据集差别不大

**本项目两种方式都用：**
- `retrieve.py` 用 FAISS（demo 用，方便）
- `evaluate.py` 用 NumPy 矩阵乘法（评估用，更灵活，不需要加载 .faiss 文件）

## 文本检索 vs 图像检索

```python
# 文本检索
tokens = clip.tokenize([query_text]).to(device)
query_emb = model.encode_text(tokens)       # (1, 512)

# 图像检索
img = preprocess(PIL_image).unsqueeze(0).to(device)
query_emb = model.encode_image(img)         # (1, 512)
```

流程完全一样，只是用了不同的 encoder。  
CLIP 保证两者在同一个空间，所以可以直接比较。

## Fashion-IQ Caption 的特殊处理

每条 query 有**两句** caption，取平均：

```python
emb1 = encode_text(model, cap1)    # (512,)
emb2 = encode_text(model, cap2)    # (512,)
text_emb = (emb1 + emb2) / 2
text_emb /= np.linalg.norm(text_emb)   # ← 必须重新归一化！
```

为什么要重新归一化：两个单位向量的平均值不再是单位向量（长度 < 1），  
不归一化的话 cosine 相似度计算就不对了。

> → 归一化语法看不懂？[concepts/normalize.md](concepts/normalize.md)

## 检索不到的情况

脚本里有大量这样的 guard：

```python
if target_asin not in asin_to_idx or candidate_asin not in asin_to_idx:
    continue
```

原因：不是所有图片都能成功下载。如果 gallery embeddings 里没有某张图，  
就跳过这条 query，最终评估结果里会显示 `Skipped N queries`。
