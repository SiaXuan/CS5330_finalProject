"""
extract_features.py — 离线提取所有 gallery 图片的 CLIP 特征

这是整个 pipeline 的第一步（Member A 负责）。
运行一次，结果保存到 features/ 目录，后续所有脚本共用。

输入:  images/dress/  目录下的所有 .jpg 图片
输出:  features/dress_embeddings.npy  — shape (N, 512)，每行一张图的向量
       features/dress_paths.txt       — 对应图片路径，行号与 embeddings 行号一一对应

详见 notes/01_clip.md    — CLIP 是什么，为什么要归一化
详见 notes/03_retrieval.md — Gallery embeddings 的作用
"""
import os
import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- 模型加载 ---
# CLIP ViT-B/32：图像 encoder，输出 512 维向量
# preprocess：把 PIL Image 转成 CLIP 需要的 224×224 tensor（含归一化）
# 详见 notes/01_clip.md — "ViT-B/32 是什么"
print("Loading CLIP model...")
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()  # 推理模式，关掉 dropout / batch norm 的训练行为

# Config
IMAGE_DIR = "images/dress"
OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 收集图片路径 ---
image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
print(f"Found {len(image_paths)} images")

# --- 逐张提取特征 ---
embeddings = []
valid_paths = []

# torch.no_grad()：推理时不需要计算梯度，节省内存和时间
with torch.no_grad():
    for path in tqdm(image_paths):
        try:
            # preprocess：resize + center crop 到 224×224，然后 normalize 成 tensor
            # unsqueeze(0)：加 batch 维，(3,224,224) → (1,3,224,224)
            image = preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
            embedding = model.encode_image(image)

            # L2 归一化：让向量长度为 1，之后 dot product = cosine 相似度
            # 详见 notes/01_clip.md — "为什么要 normalize"
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            embeddings.append(embedding.cpu().numpy()[0])  # (512,)
            valid_paths.append(path)
        except Exception:
            # 跳过损坏或读不了的图片，不中断整个流程
            continue

# --- 保存结果 ---
# embeddings: list of (512,) → stacked 成 (N, 512)
embeddings = np.array(embeddings)
np.save(os.path.join(OUTPUT_DIR, "dress_embeddings.npy"), embeddings)

# 路径文件：第 i 行对应 embeddings[i]，之后用 asin_to_idx 建索引
with open(os.path.join(OUTPUT_DIR, "dress_paths.txt"), 'w') as f:
    for p in valid_paths:
        f.write(p + '\n')

print(f"Done. Saved {len(valid_paths)} embeddings.")