import os
import clip
import faiss
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

def visualize_text_retrieval(query, top_k=5):
    """Visualize top-k retrieved images for a text query."""
    with torch.no_grad():
        text = clip.tokenize([query]).to(device)
        text_embedding = model.encode_text(text)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.cpu().numpy().astype('float32')

    scores, indices = index.search(text_embedding, top_k)

    fig, axes = plt.subplots(1, top_k, figsize=(15, 4))
    fig.suptitle(f'Text query: "{query}"', fontsize=14)
    for i, idx in enumerate(indices[0]):
        img = Image.open(image_paths[idx]).convert('RGB')
        axes[i].imshow(img)
        axes[i].set_title(f'score: {scores[0][i]:.3f}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(f'results_text_{query.replace(" ", "_")}.png')
    print(f"Saved results_text_{query.replace(' ', '_')}.png")

def visualize_image_retrieval(query_path, top_k=5):
    """Visualize top-k retrieved images for an image query."""
    with torch.no_grad():
        image = preprocess(Image.open(query_path).convert('RGB')).unsqueeze(0).to(device)
        image_embedding = model.encode_image(image)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        image_embedding = image_embedding.cpu().numpy().astype('float32')

    scores, indices = index.search(image_embedding, top_k)

    fig, axes = plt.subplots(1, top_k + 1, figsize=(18, 4))
    fig.suptitle('Image retrieval results', fontsize=14)

    # Show query image
    axes[0].imshow(Image.open(query_path).convert('RGB'))
    axes[0].set_title('Query')
    axes[0].axis('off')

    # Show results
    for i, idx in enumerate(indices[0]):
        img = Image.open(image_paths[idx]).convert('RGB')
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f'score: {scores[0][i]:.3f}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    query_name = os.path.basename(query_path).replace('.jpg', '')
    plt.savefig(f'results_image_{query_name}.png')
    print(f"Saved results_image_{query_name}.png")

# Run visualizations
visualize_text_retrieval("red dress", top_k=5)
visualize_text_retrieval("black sleeveless dress", top_k=5)
visualize_image_retrieval(image_paths[0], top_k=5)