# Glossary (with tiny code snippets)

## CLIP / OpenCLIP
**What:** Contrastive Language–Image Pretraining. Aligns text and images in the same embedding space for zero-shot retrieval.
**Why we need it:** Retrieve **non-text** regions (figures, charts) using a text query.

**Snippet:**
```python
import open_clip, torch, torch.nn.functional as F
model, preprocess, tok = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"
)
txt = tok(["evening attendance and snack sales"])
with torch.no_grad():
    t_emb = F.normalize(model.encode_text(txt), dim=-1)  # (1, D)
```

## Sentence-Transformers (MiniLM)
**What**: Lightweight text embedding model.
**Why**: Fast text cosine similarity over OCR’d patches.

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
q = m.encode(["evening attendance"], normalize_embeddings=True)[0]
d = m.encode(["attendance table row"], normalize_embeddings=True)[0]
cos = (q @ d)  # cosine similarity
```


## Tesseract OCR
**What**: System OCR engine.
**Why**: Extract text from the PDF page to feed text retrieval.
```python
from PIL import Image
import pytesseract
img = Image.open("page.png")
data = pytesseract.image_to_data(img, lang="eng", output_type=pytesseract.Output.DICT)
```

## PyMuPDF (Rasterization)
**What**: Render PDF pages to images.
**Why**: Turn manga_kissa_report.pdf into a numpy image.

```python
import fitz, numpy as np
doc = fitz.open("manga_kissa_report.pdf")
pix = doc[0].get_pixmap(dpi=300, alpha=False)
rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
```

## MobileNetV3-Small (via timm)
**What**: CPU-friendly vision backbone.
**Why**: Turn each micro-patch into a compact 256-D token.
```python
import timm, torch.nn as nn
backbone = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=0).eval()
proj = nn.Linear(backbone.num_features, 256, bias=False)
```

## Widen → Attend → Compress
**What**: Expand 256→1024 (capacity), do self-attention over 4096 tokens, compress back 1024→256.
**Why**: Long-context mixing before storing compact tokens.

```python
# single-head
import math, torch, torch.nn as nn, torch.nn.functional as F
class SA(nn.Module):
    def __init__(self, d): 
        super().__init__(); self.q=self.k=self.v=nn.Linear(d,d,bias=False); self.s=1/math.sqrt(d)
    def forward(self,x): 
        a=(self.q(x)@self.k(x).T)*self.s; a=F.softmax(a,-1); return a@self.v(x)
```

## Halo / Neighbor Awareness
**What**: Extract each micro-patch with a small overlap so it “sees” neighbors.
**Why**: Preserves local continuity (e.g., table borders, chart axes).

```python
# coordinated only
y0 = max(r*ph - HALO, 0); y1 = min((r+1)*ph + HALO, H)
x0 = max(c*pw - HALO, 0); x1 = min((c+1)*pw + HALO, W)
```

## Hybrid Retrieval Score
**What**: Combine text+vision signals.
**Why**: Queries may target text or graphics.

Formula: score = α * cosine_text + β * cosine_clip (default α=0.7, β=0.3)

```python
score = 0.7 * sims_text + 0.3 * sims_clip
topk = score.argsort()[::-1][:3]
```

## Heatmap & Top-K Boxes
**What**: Visual proof of where the model “looks.”
**Why**: Great for demos; shows that best score matches the right region.

```python
import cv2, numpy as np
heat = (score - score.min()) / (score.max() - score.min() + 1e-9)
heat = cv2.resize(heat.reshape(GRID, GRID), (W, H), cv2.INTER_CUBIC)
overlay = cv2.addWeighted(page_bgr, 1.0, cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET), 0.45, 0)
```

## @ Operator (Matrix Multiplication)
**What it is**:
In Python, the `@` symbol is the matrix multiplication operator.
It was introduced in `Python3.5` (`PEP 465`) to make linear-algebra expressions cleaner and easier to read.

**Why we use it in `ML`**:
Machine-learning models—especially attention mechanisms—do tons of matrix multiplications:
- combining embeddings
- computing attention scores (q @ k.T)
- projecting data through linear layers

So `@` is just a shortcut for `numpy.matmul()` or `torch.matmul()`, depending on the library.
Example:
```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])     # shape (2×2)
B = np.array([[5, 6],
              [7, 8]])     # shape (2×2)

# Same result:
C1 = A @ B                 # matrix multiplication
C2 = np.matmul(A, B)       # equivalent

print(C1)
# [[19 22]
#  [43 50]]
```

In our `Creditzens-DeepSeek-OCR Experiement`:
```python
att = (q @ k.T) * scale   # multiply query and key matrices
att = softmax(att)        # attention weights
output = att @ v          # combine with value matrix
```

Here `@` performs dot products across all tokens—the core of the attention step that lets the model mix information globally across the 4096 patches.
So in short:
- `@` → "matrix multiply."
It replaces long `torch.matmul()` calls and makes attention equations readable, like in math papers.
