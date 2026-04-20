import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def load_model(model_name: str):
    if model_name == "clip":
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        return model, preprocess
    elif model_name == "fashionclip":
        from fashion_clip.fashion_clip import FashionCLIP
        model = FashionCLIP('fashion-clip')
        return model, None

def encode_images_clip(model, preprocess, image_paths):
    embeddings, valid_paths = [], []
    with torch.no_grad():
        for path in tqdm(image_paths):
            try:
                image = preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
                emb = model.encode_image(image)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().numpy()[0])
                valid_paths.append(path)
            except Exception:
                raise
    return np.array(embeddings), valid_paths

def encode_images_fashionclip(model, image_paths, batch_size=32):
    processor = model.preprocess
    vision_model = model.model.vision_model
    dev = model.device

    embeddings, valid_paths = [], []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i : i + batch_size]
        try:
            images = [Image.open(p).convert('RGB') for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(dev)
            with torch.no_grad():
                emb = vision_model(pixel_values=pixel_values).pooler_output  # (B, 768)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy().astype("float32"))
            valid_paths.extend(batch_paths)
        except Exception:
            raise
    return np.concatenate(embeddings, axis=0), valid_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["clip", "fashionclip"], default="clip")
    parser.add_argument("--category", default="dress")
    args = parser.parse_args()

    image_dir = f"images/{args.category}"
    output_dir = "features"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir) if f.endswith('.jpg')
    ])
    print(f"Found {len(image_paths)} images in {image_dir}")

    print(f"Loading {args.model}...")
    model, preprocess = load_model(args.model)

    print("Extracting features...")
    if args.model == "clip":
        embeddings, valid_paths = encode_images_clip(model, preprocess, image_paths)
    else:
        embeddings, valid_paths = encode_images_fashionclip(model, image_paths)

    emb_path = os.path.join(output_dir, f"{args.category}_embeddings_{args.model}.npy")
    paths_path = os.path.join(output_dir, f"{args.category}_paths_{args.model}.txt")

    np.save(emb_path, embeddings)
    with open(paths_path, 'w') as f:
        for p in valid_paths:
            f.write(p + '\n')

    print(f"Done. Saved {len(valid_paths)} embeddings → {emb_path}")

if __name__ == "__main__":
    main()
