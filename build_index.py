import faiss
import numpy as np

# Load embeddings
print("Loading embeddings...")
embeddings = np.load("features/dress_embeddings.npy").astype('float32')
print(f"Embeddings shape: {embeddings.shape}")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity (embeddings are normalized)
index.add(embeddings)
print(f"Index built with {index.ntotal} vectors")

# Save index
faiss.write_index(index, "features/dress_index.faiss")
print("Index saved to features/dress_index.faiss")