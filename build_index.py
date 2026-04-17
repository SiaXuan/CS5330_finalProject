"""
build_index.py — 把 gallery embeddings 建成 FAISS 向量索引

第二步：extract_features.py 跑完之后运行这个。
FAISS 索引是 retrieve.py 和 text_retrieval.py 的前置条件；
evaluate.py 直接用 NumPy 矩阵乘法，不依赖这个文件。

详见 notes/03_retrieval.md — "FAISS：IndexFlatIP"
"""
import faiss
import numpy as np

print("Loading embeddings...")
# float32 是 FAISS 要求的精度
embeddings = np.load("features/dress_embeddings.npy").astype('float32')
print(f"Embeddings shape: {embeddings.shape}")  # 期望: (N, 512)

# --- 建 FAISS 内积索引 ---
# IndexFlatIP = 穷举内积（Inner Product）搜索
# 因为 embeddings 已归一化，内积 = cosine 相似度
# 详见 notes/03_retrieval.md — "为什么用矩阵乘法"
dimension = embeddings.shape[1]  # 512
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"Index built with {index.ntotal} vectors")

faiss.write_index(index, "features/dress_index.faiss")
print("Index saved to features/dress_index.faiss")