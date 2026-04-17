"""
retrieve.py — Member A 的基础检索 demo（用 FAISS 索引）

这是最简单的版本，展示了两个检索函数：
  text_retrieval(query)        : 文字 → top-k 图片
  image_retrieval(image_path)  : 图片 → top-k 图片

注意：这个脚本依赖 FAISS 索引（build_index.py 的输出）。
Member B 的 text_retrieval.py 和 Member C 的 evaluate.py
不用 FAISS，改用 NumPy 矩阵乘法，更灵活。

详见 notes/03_retrieval.md — 完整检索原理
"""
import os
import clip
import faiss
import torch
import numpy as np
from PIL import Image

# 加载模型和索引（脚本导入时就跑，注意是 module-level 代码）
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# FAISS 索引 + 对应图片路径
# 路径列表的行号 i 对应 gallery_embeddings[i]
index = faiss.read_index("features/dress_index.faiss")
with open("features/dress_paths.txt") as f:
    image_paths = [line.strip() for line in f.readlines()]


def text_retrieval(query, top_k=5):
    """文字 → top-k 图片（用 CLIP text encoder + FAISS 搜索）"""
    with torch.no_grad():
        text = clip.tokenize([query]).to(device)
        text_embedding = model.encode_text(text)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.cpu().numpy().astype('float32')

    # index.search 返回 (scores, indices)，形状都是 (1, top_k)
    scores, indices = index.search(text_embedding, top_k)
    results = [(image_paths[i], scores[0][j]) for j, i in enumerate(indices[0])]
    return results


def image_retrieval(query_image_path, top_k=5):
    """图片 → top-k 图片（用 CLIP image encoder + FAISS 搜索）"""
    with torch.no_grad():
        image = preprocess(Image.open(query_image_path).convert('RGB')).unsqueeze(0).to(device)
        image_embedding = model.encode_image(image)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        image_embedding = image_embedding.cpu().numpy().astype('float32')

    scores, indices = index.search(image_embedding, top_k)
    results = [(image_paths[i], scores[0][j]) for j, i in enumerate(indices[0])]
    return results


# --- 简单 demo ---
print("Testing text retrieval...")
query = "red dress"
results = text_retrieval(query, top_k=5)
print(f"Query: '{query}'")
for path, score in results:
    print(f"  {os.path.basename(path)}  score: {score:.4f}")

print("\nTesting image retrieval...")
sample_image = image_paths[0]
results = image_retrieval(sample_image, top_k=5)
print(f"Query image: {os.path.basename(sample_image)}")
for path, score in results:
    print(f"  {os.path.basename(path)}  score: {score:.4f}")