import os
import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Load CLIP model
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Config
IMAGE_DIR = "images/dress"
OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all image paths
image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
print(f"Found {len(image_paths)} images")

# Extract features
embeddings = []
valid_paths = []

with torch.no_grad():
    for path in tqdm(image_paths):
        try:
            image = preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
            embedding = model.encode_image(image)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # normalize
            embeddings.append(embedding.cpu().numpy()[0])
            valid_paths.append(path)
        except Exception:
            continue

# Save embeddings and paths
embeddings = np.array(embeddings)
np.save(os.path.join(OUTPUT_DIR, "dress_embeddings.npy"), embeddings)

with open(os.path.join(OUTPUT_DIR, "dress_paths.txt"), 'w') as f:
    for p in valid_paths:
        f.write(p + '\n')

print(f"Done. Saved {len(valid_paths)} embeddings.")