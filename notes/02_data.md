# Fashion-IQ 数据格式

> 相关代码：`text_retrieval.py`, `evaluate.py`, `compare_viz.py`

## 数据集的任务定义

Fashion-IQ 是一个交互式时尚图像检索数据集。  
每条 query 代表：

> "我看到了这件衣服（candidate），想找那件（target），区别是……（captions）"

所以这个数据集天然就是一个**多模态 query**：图片 + 文字描述 → 找目标图片。

## 目录结构

```
fashion-iq/
├── captions/
│   ├── cap.dress.train.json   ← 训练集（~6000 条 query）
│   ├── cap.dress.val.json     ← 验证集（~2000 条 query）← 我们主要用这个
│   └── cap.dress.test.json
│
└── image_splits/
    ├── split.dress.train.json ← 训练集图片 ASIN 列表
    ├── split.dress.val.json   ← 验证集图片 ASIN 列表 ← gallery（检索图库）
    └── split.dress.test.json
```

类别除了 `dress` 还有 `shirt` 和 `toptee`，结构相同。

## Caption 文件格式

文件：`fashion-iq/captions/cap.dress.val.json`

```json
[
  {
    "target":    "B008BHCT58",
    "candidate": "B003FGW7MK",
    "captions":  [
      "is solid black with no sleeves",
      "is black with straps"
    ]
  },
  ...
]
```

**字段说明：**

| 字段 | 含义 |
|------|------|
| `target` | 目标图片的 ASIN（我们要检索出来的那张图） |
| `candidate` | 参考图片的 ASIN（query 起点，"我现在看的这件"） |
| `captions` | 两句描述 target 相对 candidate 的变化，由不同标注人写的 |

**注意：** captions 描述的是**差异**，不是 target 的完整描述。  
例如 "is solid black with no sleeves" 的意思是 target 是纯黑无袖的（相对于 candidate）。

## Image Split 文件格式

文件：`fashion-iq/image_splits/split.dress.val.json`

```json
["B008BHCT58", "B00CHSEGYE", "B00BI588U0", "B00LLGE2SK", ...]
```

这个列表里的所有 ASIN 就是检索的 **gallery**（待检索图库）。

## ASIN 是什么

Amazon Standard Identification Number — 亚马逊商品唯一 ID。  
图片下载后存为 `images/dress/{ASIN}.jpg`。  
脚本里经常从路径中提取 ASIN：

```python
asin = os.path.splitext(os.path.basename(path))[0]
# "images/dress/B008BHCT58.jpg" → "B008BHCT58"
```

## 三个概念的关系

```
Gallery = split.dress.val.json 里的所有图片（~4000 张）
                    ↑
Query = cap.dress.val.json 里每一条 (candidate, captions) 对（~2000 条）
                    ↓
Target = 每条 query 对应的目标图片（在 gallery 里）
```

**评估时：** 对每条 query，在整个 gallery 里检索，看 target 排第几位。

## 数量统计（dress 类）

| 集合 | Query 数 | Gallery 大小 |
|------|----------|--------------|
| val  | ~2,000   | ~4,000 张    |
| train| ~6,000   | ~12,000 张   |

> 我们只用 val 集做评估，train 集可以用来调参（本项目不需要）。

## 为什么 candidate 和 target 都在 gallery 里

`cap.dress.val.json` 的 candidate 和 target ASIN 都来自 `split.dress.val.json`。  
这意味着：
- candidate 的 embedding 可以直接从预计算的 gallery embeddings 里查到
- 不需要额外读取图片文件，速度更快

`evaluate.py` 利用了这一点：
```python
cand_idx  = asin_to_idx[entry["candidate"]]
image_emb = gallery_emb[cand_idx]  # 直接查，不读硬盘
```
