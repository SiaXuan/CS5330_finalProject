import os
import clip
import faiss
import torch
import numpy as np
from PIL import Image

# Load CLIP model
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Load index and paths
index = faiss.read_index("features/dress_index.faiss")
with open("features/dress_paths.txt") as f:
    image_paths = [line.strip() for line in f.readlines()]

def text_retrieval(query, top_k=5):
    """Retrieve top-k images given a text query."""
    with torch.no_grad():
        text = clip.tokenize([query]).to(device)
        text_embedding = model.encode_text(text)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.cpu().numpy().astype('float32')

    scores, indices = index.search(text_embedding, top_k)
    results = [(image_paths[i], scores[0][j]) for j, i in enumerate(indices[0])]
    return results

def image_retrieval(query_image_path, top_k=5):
    """Retrieve top-k images given a query image."""
    with torch.no_grad():
        image = preprocess(Image.open(query_image_path).convert('RGB')).unsqueeze(0).to(device)
        image_embedding = model.encode_image(image)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        image_embedding = image_embedding.cpu().numpy().astype('float32')

    scores, indices = index.search(image_embedding, top_k)
    results = [(image_paths[i], scores[0][j]) for j, i in enumerate(indices[0])]
    return results

# Test text retrieval
print("Testing text retrieval...")
query = "red dress"
results = text_retrieval(query, top_k=5)
print(f"Query: '{query}'")
for path, score in results:
    print(f"  {os.path.basename(path)}  score: {score:.4f}")

# Test image retrieval
print("\nTesting image retrieval...")
sample_image = image_paths[0]
results = image_retrieval(sample_image, top_k=5)
print(f"Query image: {os.path.basename(sample_image)}")
for path, score in results:
    print(f"  {os.path.basename(path)}  score: {score:.4f}")