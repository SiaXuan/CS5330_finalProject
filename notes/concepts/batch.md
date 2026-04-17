# 概念：batch 和 unsqueeze(0)

## 什么是 batch

深度学习模型通常**批量**处理数据，一次输入多个样本，叫做一个 batch。

```python
# 一次处理 32 张图：shape (32, 3, 224, 224)
# 一次处理 64 条文字：shape (64, 77)
```

PyTorch 的模型统一要求输入的**第一个维度是 batch 大小**，
哪怕只处理 1 个样本，也要保留这一层，变成 batch_size=1。

## unsqueeze(0) 做了什么

`unsqueeze(n)` = 在第 n 个位置插入一个新维度（大小为 1）。

```python
x = torch.tensor([1, 2, 3])   # shape: (3,)
x.unsqueeze(0)                 # shape: (1, 3)  ← 在最前面加一层
x.unsqueeze(1)                 # shape: (3, 1)  ← 在中间加一层
```

在图像处理里：

```python
image = preprocess(img)         # shape: (3, 224, 224)   C × H × W
image = image.unsqueeze(0)      # shape: (1, 3, 224, 224) batch × C × H × W
```

加了这一层，模型才能接受（它期望的格式是 batch × ...）。

## 处理完之后去掉这一层

模型输出也带着 batch 维：

```python
emb = model.encode_image(image)   # shape: (1, 512)
emb = emb[0]                      # shape: (512,)  ← 取第 0 个样本，去掉 batch 层
# 或者
emb.cpu().numpy()[0]              # 同上，同时转成 numpy array
```

## 为什么 gallery 不需要 unsqueeze

`extract_features.py` 里每张图单独处理，用了 `unsqueeze(0)` 变成 `(1, 512)`，
取结果时用 `[0]` 去掉 batch 层拿到 `(512,)`，然后 append 进列表。

最后 `np.array(embeddings)` 把列表 stack 成 `(N, 512)`，
N 张图各自一行，就是 gallery embeddings。

```python
embeddings.append(embedding.cpu().numpy()[0])  # 每次存 (512,)
# ...
np.array(embeddings)  # stack → (N, 512)
```
