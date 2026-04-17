# CLIP 模型

> 相关代码：`extract_features.py`, `text_retrieval.py`, `evaluate.py`

## 是什么

CLIP（Contrastive Language-Image Pre-training，OpenAI 2021）是一个**同时理解图像和文字**的预训练模型。

**训练方式：** 用 4 亿个（图片, 文字描述）对做训练，让"一只红色的猫"的文字向量和猫图片的向量尽量接近，和其他无关图片的向量尽量远离。这叫**对比学习**。

**效果：** CLIP 学到了一个共同的向量空间，图像和文字可以直接比较相似度，不需要任何额外标注数据。

## 我们用的版本：ViT-B/32

```python
model, preprocess = clip.load("ViT-B/32", device=device)
```

- `ViT` = Vision Transformer：把图片切成 32×32 的小块（patch），当 token 送进 Transformer 处理
- `B` = Base 版（中等大小，还有更大的 L、H 版，但我们不需要）
- `/32` = 每个 patch 是 32×32 像素
- 输出一个 **512 维**的向量

## 两个 Encoder

```python
# 图像 → 向量
image_tensor = preprocess(PIL_image).unsqueeze(0).to(device)
image_emb = model.encode_image(image_tensor)   # shape: (1, 512)

# 文字 → 向量
tokens = clip.tokenize(["red dress"]).to(device)
text_emb = model.encode_text(tokens)            # shape: (1, 512)
```

> → 不懂 `shape: (1, 512)`？[concepts/shape.md](concepts/shape.md)  
> → 不懂 `unsqueeze(0)`？[concepts/batch.md](concepts/batch.md)

两者输出都在**同一个 512 维空间**，所以可以直接做 dot product 计算相似度。

## 为什么要 normalize（L2 归一化）

```python
emb = emb / emb.norm(dim=-1, keepdim=True)
```

> → 不懂这行代码的语法？[concepts/normalize.md](concepts/normalize.md)

向量归一化之后，内积（dot product）= cosine 相似度，值域 [-1, 1]：

- `1.0` → 完全相同方向（最相似）
- `0.0` → 无关
- `-1.0` → 完全相反

归一化后用 FAISS `IndexFlatIP`（内积索引）等价于 cosine 检索。

**如果不归一化，** 向量长度会影响分数，同一图片在不同 batch 下分数可能不同，评估结果不可靠。

## clip.tokenize 的 truncate=True

```python
tokens = clip.tokenize([query], truncate=True)
```

CLIP 的文字 encoder 最多接受 77 个 token（约 50 个英文单词）。  
`truncate=True` 表示超出部分截断，否则会报错。  
Fashion-IQ 的 caption 通常很短，不会触发这个限制，但加上是好习惯。

## 为什么 CLIP 适合时尚检索

1. **Zero-shot**：不需要训练，直接用预训练权重
2. **视觉-语义对齐**：能理解"黑色"、"无袖"、"蕾丝"等时尚属性
3. **统一空间**：图文可以直接比较，天然支持跨模态检索
4. **轻量**：ViT-B/32 推理很快，不需要 GPU 也能跑
