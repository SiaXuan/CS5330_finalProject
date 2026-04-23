"""
Fashion Search Demo — Flask backend

Run from the project root:
    source venv/bin/activate
    python demo/app.py

Then open http://localhost:5000
"""
import os
import sys
import numpy as np
import torch
from flask import Flask, request, jsonify, send_file, render_template
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tirg.model import TIRGCombiner

CATEGORIES  = ["dress", "shirt", "toptee"]
FEATURES_DIR = "features"
CHECKPOINT   = "tirg/checkpoints/tirg_all_best.pt"
TOP_K        = 48   # fetch more; frontend paginates

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── globals loaded at startup ─────────────────────────────────────────────────
gallery_embs  = None   # (N, 512) float32
gallery_asins = None   # list[str]
gallery_paths = None   # list[str]
gallery_cats  = None   # list[str]
asin_to_idx   = None   # dict[str, int]
fc_model      = None
combiner      = None


def load_gallery():
    global gallery_embs, gallery_asins, gallery_paths, gallery_cats, asin_to_idx
    all_embs, all_asins, all_paths, all_cats = [], [], [], []
    for cat in CATEGORIES:
        embs = np.load(f"{FEATURES_DIR}/{cat}_embeddings_fashionclip.npy").astype("float32")
        with open(f"{FEATURES_DIR}/{cat}_paths_fashionclip.txt") as f:
            paths = [line.strip() for line in f]
        for path, emb in zip(paths, embs):
            all_embs.append(emb)
            all_paths.append(path)
            all_asins.append(Path(path).stem.strip())
            all_cats.append(cat)
    # Only keep images that were actually downloaded
    exists = [os.path.exists(p) for p in all_paths]
    all_embs  = [e for e, ok in zip(all_embs,  exists) if ok]
    all_paths = [p for p, ok in zip(all_paths, exists) if ok]
    all_asins = [a for a, ok in zip(all_asins, exists) if ok]
    all_cats  = [c for c, ok in zip(all_cats,  exists) if ok]

    gallery_embs  = np.stack(all_embs)
    gallery_asins = all_asins
    gallery_paths = all_paths
    gallery_cats  = all_cats
    asin_to_idx   = {a: i for i, a in enumerate(all_asins)}
    print(f"Gallery: {len(all_asins)} images (with files) across {len(CATEGORIES)} categories")


def encode_text(text: str) -> np.ndarray:
    processor  = fc_model.preprocess
    text_model = fc_model.model.text_model
    dev        = fc_model.device
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        pooled = text_model(
            input_ids      = inputs["input_ids"].to(dev),
            attention_mask = inputs["attention_mask"].to(dev),
        ).pooler_output
        emb = fc_model.model.text_projection(pooled)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].astype("float32")


def filtered_search(query_emb: np.ndarray, category: str | None,
                    exclude_asin: str | None = None) -> list[dict]:
    if category and category in CATEGORIES:
        mask = np.array([c == category for c in gallery_cats])
    else:
        mask = np.ones(len(gallery_cats), dtype=bool)

    if exclude_asin and exclude_asin in asin_to_idx:
        mask[asin_to_idx[exclude_asin]] = False

    indices = np.where(mask)[0]
    scores  = gallery_embs[indices] @ query_emb
    top_local = np.argsort(scores)[::-1][:TOP_K]

    results = []
    for li in top_local:
        gi = indices[li]
        results.append({
            "asin":     gallery_asins[gi],
            "category": gallery_cats[gi],
            "score":    float(scores[li]),
        })
    return results


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/image/<asin>")
def serve_image(asin):
    idx = asin_to_idx.get(asin)
    if idx is None:
        return "Not found", 404
    path = os.path.abspath(gallery_paths[idx])
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/api/search", methods=["POST"])
def search():
    data     = request.json
    text     = data.get("text", "").strip()
    category = data.get("category") or None
    if not text:
        return jsonify({"error": "text required"}), 400

    query_emb = encode_text(text)
    results   = filtered_search(query_emb, category)
    return jsonify({"results": results})


@app.route("/api/refine", methods=["POST"])
def refine():
    data      = request.json
    ref_asin  = data.get("reference_asin", "").strip()
    text      = data.get("text", "").strip()
    category  = data.get("category") or None
    if not ref_asin or not text:
        return jsonify({"error": "reference_asin and text required"}), 400

    ref_idx = asin_to_idx.get(ref_asin)
    if ref_idx is None:
        return jsonify({"error": "Reference image not in gallery"}), 404

    ref_emb  = torch.tensor(gallery_embs[ref_idx]).unsqueeze(0).to(device)
    txt_emb  = torch.tensor(encode_text(text)).unsqueeze(0).to(device)
    with torch.no_grad():
        tirg_emb = combiner(ref_emb, txt_emb).cpu().numpy()[0]

    # Blend TIRG output with the original image embedding so visual attributes
    # (e.g. floral pattern) aren't washed out when the modification text only
    # describes one dimension of change.
    ref_np    = gallery_embs[ref_idx]
    combined  = 0.6 * tirg_emb + 0.4 * ref_np
    query_emb = (combined / np.linalg.norm(combined)).astype("float32")

    # Default to same category as reference if none specified
    if not category:
        category = gallery_cats[ref_idx]

    results = filtered_search(query_emb, category, exclude_asin=ref_asin)
    return jsonify({"results": results})


# ── startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading FashionCLIP...")
    from fashion_clip.fashion_clip import FashionCLIP
    fc_model = FashionCLIP("fashion-clip")

    print("Loading TIRG combiner...")
    combiner = TIRGCombiner(feature_dim=512).to(device)
    combiner.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    combiner.eval()

    print("Loading gallery...")
    load_gallery()

    print("\nReady → http://localhost:5000\n")
    app.run(debug=False, port=8080)
