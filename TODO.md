# TODO

## 支持库外 query 图片（open-set retrieval）

**现状：** `evaluate.py` 和 `fusion_retrieval.py` 的 image query
都假设 candidate 图片在 gallery 里，直接查预计算的向量。

**目标：** 允许传入任意图片路径作为 image query，不要求它在 gallery 里。

**需要改的地方：**

- `fusion_retrieval.py` — `get_query_embeddings()` 里把 gallery 切片改成 `encode_image()`。
  改动量很小，函数已经存在。

- `evaluate.py` — `evaluate_all()` 里目前用 `gallery_emb[cand_idx]` 批量取向量。
  改成对所有 candidate 图片跑 batch encode（类似 `encode_text_batch` 的写法）。
  会慢几分钟，但结构不变。

**实际价值：** 改完之后可以直接拿手机拍的衣服图片作为 query 检索 gallery，
变成真正可演示的检索系统。
