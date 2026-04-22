"""Model-agnostic re-ranker for Fashion-IQ retrieval.

The whole package consumes only numpy embedding arrays — it never imports
CLIP or FashionCLIP. That keeps it swap-safe: whatever encoder produced
`features/*_embeddings.npy` and the text embeddings from
`prepare_embeddings.py`, this package will re-rank its top-K.
"""
