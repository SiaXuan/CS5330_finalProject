import os
import json
import numpy as np

# Image stats
image_dir = "images/dress"
total_images = len([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

# Embedding stats
embeddings = np.load("features/dress_embeddings.npy")

# Caption stats
caption_files = [
    "fashion-iq/captions/cap.dress.train.json",
    "fashion-iq/captions/cap.dress.val.json",
]
total_captions = 0
for cf in caption_files:
    with open(cf) as f:
        total_captions += len(json.load(f))

print("=== Data Summary ===")
print(f"Total images in gallery:     {total_images}")
print(f"Total embeddings extracted:  {embeddings.shape[0]}")
print(f"Embedding dimension:         {embeddings.shape[1]}")
print(f"Total caption pairs:         {total_captions}")
print(f"Categories:                  dress, shirt (mixed)")