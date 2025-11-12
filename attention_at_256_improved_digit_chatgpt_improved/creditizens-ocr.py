"""
demo.py — CPU-only, 1-page DeepSeek-OCR mimic for @creditizens

What it shows (closer to the paper's Figure 3):
1) PDF -> image
2) Window-attention tokenizer (Swin-Tiny) at stride ~16 -> vision tokens on a 2D grid
3) Project channels -> 256-D compact token width
4) 16× token compressor (sequence) via 2D average pooling (/4 x /4) -> far fewer tokens
5) Global self-attention over the compressed sequence at 256-D (long context, low width)
6) Save compact tokens
7) OCR page, map lines to compressed token grid (rows×cols)
8) Dual retrieval (Text via MiniLM, Vision via OpenCLIP on compressed cells) + hybrid scoring
9) Top-k results + heatmap visualization (rows×cols grid)
10) (Optional) Local Ollama agent to synthesize a friendly answer

CPU friendly (one page). No training — architecture demo.
"""

"""
# 1) Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# 2) Start the server (new terminal or background)
ollama serve

# 3) Pull a small, sharp model (pick ONE of these)
ollama pull qwen2.5:1.5b-instruct
# or
ollama pull llama3.2:1b-instruct
# or
ollama pull phi3:mini

# 4) Ensure 'requests' is installed for the script
pip install requests

# 5) Run your pipeline + agent
python3 creditizens-ocr.py \
  --pdf ../manga_kissa_report.pdf \
  --query "evening attendance and snack sales" \
  --beta 0.3 \
  --ollama \
  --ollama-model qwen2.5:1.5b-instruct \
  --ollama-num-ctx 4096 \
  --ollama-num-predict 512 \
  --ollama-topn-cells 300 \
  --out-prefix results_topk \
  --confidence-score 54
# can add `--tokenizer-max-side` for limit cap on memory usage
#  --tokenizer-max-side 768
"""

# --- Project-local cache + stable threading (very top, before ALL imports) ---
import os
HF_HOME = os.path.join(os.getcwd(), ".hf_cache")
os.makedirs(HF_HOME, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_HOME)
os.environ.setdefault("TORCH_HOME", os.path.join(HF_HOME, "torch"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
  import torch
  torch.set_num_threads(1)
except Exception:
  pass

# stdlib and libs
import json, math, argparse, re, time, textwrap
import numpy as np
import cv2
from PIL import Image
import fitz
import pytesseract
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sentence_transformers import SentenceTransformer
import open_clip

# optional dependency for Ollama HTTP calls
try:
  import requests
except Exception:
  requests = None

# --------------------------- CLI & Defaults ---------------------------

def parse_args():
  # argument parser
  ap = argparse.ArgumentParser(description="DeepSeek-OCR mimic (CPU-only, 1-page)")
  # input pdf and raster dpi
  ap.add_argument("--pdf", required=True, help="Input 1-page PDF path")
  ap.add_argument("--dpi", type=int, default=300, help="Raster DPI (300-400 is good)")
  # legacy knobs kept for compatibility (not used in the Swin path but harmless)
  ap.add_argument("--macro-grid", type=int, default=16, help="(compat) kept for old path; unused in Swin tokenizer")
  ap.add_argument("--encoder-size", type=int, default=160, help="(compat) unused in Swin tokenizer")
  ap.add_argument("--feature_dim", type=int, default=256, help="Compact token dim (width)")
  ap.add_argument("--widen-dim", type=int, default=1024, help="(compat) unused in Swin path; width stays 256")
  # ocr and query
  ap.add_argument("--ocr-lang", default="eng", help="Tesseract language (e.g., eng, eng+fra)")
  ap.add_argument("--confidence-score", type=int, default=60, help="Confidence score threshold to filter OCR noise (0..100)")
  ap.add_argument("--query", required=True, help="User query for retrieval")
  ap.add_argument("--topk", type=int, default=3, help="Top-k results to show")
  # retrieval mixing weights
  ap.add_argument("--alpha", type=float, default=0.7, help="Weight for text score")
  ap.add_argument("--beta", type=float, default=0.3, help="Weight for CLIP vision score (set 0.0 to skip CLIP)")
  ap.add_argument("--no-clip", action="store_true", help="Force skip CLIP (debug)")
  # outputs and options
  ap.add_argument("--out-prefix", default="results_topk", help="Basename for result files")
  # ollama agent options
  ap.add_argument("--ollama", action="store_true", help="Use a local Ollama model to synthesize a final human answer")
  ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
  ap.add_argument("--ollama-model", default="qwen2.5:1.5b-instruct", help="Ollama model name (CPU-friendly)")
  ap.add_argument("--ollama-num-predict", type=int, default=512, help="Max tokens to generate")
  ap.add_argument("--ollama-num-ctx", type=int, default=4096, help="Context window size")
  ap.add_argument("--ollama-temperature", type=float, default=0.2, help="Generation temperature")
  ap.add_argument("--ollama-topn-cells", type=int, default=60, help="How many high-score cells to include in agent context")
  # tokenizer target size (we will pad up to multiples of 32 after resize)
  ap.add_argument("--tokenizer-max-side", type=int, default=896, help="Resize page so max(H,W)=this, then PAD to multiples of 32")
  return ap.parse_args()

# --------------------------- Step 1: PDF -> Image ---------------------------

def pdf_to_image(pdf_path: str, dpi: int = 300) -> np.ndarray:
  # open pdf and rasterize first page to rgb numpy array
  doc = fitz.open(pdf_path)
  page = doc[0]
  pix = page.get_pixmap(dpi=dpi, alpha=False)
  img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
  return img

# --------------------------- Grid helpers (rows × cols) ---------------------------

def make_grid_boxes_rc(H: int, W: int, rows: int, cols: int):
  # build (rows*cols, 4) boxes over the page for visualization + CLIP crops
  cell_h, cell_w = H // rows, W // cols
  boxes = []
  for r in range(rows):
    for c in range(cols):
      y0 = r * cell_h
      y1 = (r + 1) * cell_h
      x0 = c * cell_w
      x1 = (c + 1) * cell_w
      boxes.append((y0, y1, x0, x1))
  return np.array(boxes, dtype=np.int32)

# --------------------------- Swin tokenizer (local/window attention) ---------------------------

def _pad_to_multiple_of_32(img: np.ndarray):
  # compute minimal padding so H and W are multiples of 32 (required by Swin patch merging)
  H, W = img.shape[:2]
  pad_h = (32 - (H % 32)) % 32
  pad_w = (32 - (W % 32)) % 32
  if pad_h == 0 and pad_w == 0:
    return img, H, W
  # replicate border pixels to avoid artifacts
  img2 = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
  Hp, Wp = img2.shape[:2]
  return img2, Hp, Wp

def swin_tokens_from_image(image_rgb: np.ndarray, target_size=896, proj_dim=256, device="cpu"):
  # resize page so Swin’s internal windows are stable; then PAD to multiples of 32; then create Swin WITH that exact padded size
  H, W = image_rgb.shape[:2]
  scale = target_size / max(H, W)
  newH, newW = int(round(H * scale)), int(round(W * scale))
  resized = cv2.resize(image_rgb, (newW, newH), interpolation=cv2.INTER_AREA)

  # pad to multiples of 32 to satisfy Swin patch merging (patch=4, 3 merges => /32)
  padded, Hp, Wp = _pad_to_multiple_of_32(resized)

  # create Swin-Tiny with features_only=True and the exact img_size=(Hp, Wp)
  swin = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    features_only=True,
    img_size=(Hp, Wp)
  ).eval().to(device)

  # to tensor
  x = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float() / 255.0
  x = x.to(device)

  # forward through Swin and grab stride~16 feature map (stage 3)
  with torch.no_grad():
    feats = swin(x)               # list of maps at strides ~ [4, 8, 16, 32]
    f = feats[2]                  # (B, C, h, w) at stride ~16

  # flatten tokens and project channels to 256-D compact width
  B, C, h, w = f.shape
  f = f.permute(0, 2, 3, 1).reshape(B, h * w, C).squeeze(0)   # (N=h*w, C)
  proj = nn.Linear(f.shape[-1], proj_dim, bias=False).to(device)
  with torch.no_grad():
    tokens_256 = proj(f.to(device)).cpu()                      # (N, 256)

  # we return scale relative to original (pre-pad) size; padding is internal to tokenizer only
  return tokens_256, (h, w), (Hp, Wp), scale

# --------------------------- 16× token compressor (sequence) ---------------------------

def compress_tokens_2d(tokens_256: torch.Tensor, hw: tuple[int, int], pool_kernel=4):
  # reshape tokens back to grid, average-pool /4 in both dims -> ~16x fewer tokens
  h, w = hw
  t = tokens_256.reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0)   # (1, 256, h, w)
  pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=False)
  with torch.no_grad():
    t_small = pool(t)                                             # (1, 256, h/4, w/4)
  _, C, hs, ws = t_small.shape
  out = t_small.permute(0, 2, 3, 1).reshape(hs * ws, C).contiguous()  # (Nc, 256)
  return out, (hs, ws)

# --------------------------- Global attention (256-D) ---------------------------

class SimpleSelfAttention(nn.Module):
  # single-head self-attention over tokens (N, D)
  def __init__(self, dim: int, qkv_dim: int | None = None):
    super().__init__()
    qkv_dim = qkv_dim or dim
    self.q = nn.Linear(dim, qkv_dim, bias=False)
    self.k = nn.Linear(dim, qkv_dim, bias=False)
    self.v = nn.Linear(dim, qkv_dim, bias=False)
    self.scale = 1.0 / math.sqrt(qkv_dim)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    q = self.q(x)
    k = self.k(x)
    v = self.v(x)
    att = (q @ k.T) * self.scale
    att = F.softmax(att, dim=-1)
    return att @ v

# --------------------------- Save tokens ---------------------------

def save_tokens(tokens_256: np.ndarray, boxes: np.ndarray, rows: int, cols: int,
                H: int, W: int, dpi: int, out="page_tokens_compact.npz"):
  # persist compact tokens and grid meta
  np.savez(out,
           tokens=tokens_256.astype(np.float32),
           boxes=boxes,
           grid=np.array([rows, cols], dtype=np.int32),
           meta=np.array([H, W, dpi], dtype=np.int32))
  print(f"[OK] Saved tokens -> {os.path.abspath(out)}")

# --------------------------- OCR page & map lines to rows×cols ---------------------------

def _clean_text(s: str) -> str:
  # normalize spaces and trim stray punctuation
  s = re.sub(r"\s+", " ", s)
  s = s.strip(" -:;,.")
  return s.strip()

def ocr_page_to_patch_texts_rc(img_rgb: np.ndarray, rows: int, cols: int, confidence: int, lang: str = "eng") -> list[str]:
  # run tesseract and map each line center to a grid cell (rows×cols)
  H, W = img_rgb.shape[:2]
  cell_h, cell_w = H // rows, W // cols
  pil = Image.fromarray(img_rgb)
  ocr = pytesseract.image_to_data(
    pil, lang=lang, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT
  )

  # group tokens by line id
  by_line = {}
  n = len(ocr["text"])
  for i in range(n):
    txt = (ocr["text"][i] or "").strip()
    if not txt:
      continue
    try:
      conf = int(ocr["conf"][i])
    except Exception:
      conf = -1
    # confidence gate
    if conf < confidence:
      continue
    line_id = (ocr["block_num"][i], ocr["par_num"][i], ocr["line_num"][i])
    x, y, w, h = ocr["left"][i], ocr["top"][i], ocr["width"][i], ocr["height"][i]
    cx, cy = x + w // 2, y + h // 2
    if line_id not in by_line:
      by_line[line_id] = {"tokens": [], "centers": []}
    by_line[line_id]["tokens"].append(txt)
    by_line[line_id]["centers"].append((cx, cy))

  # build one string per line and drop it into its representative cell
  texts = [""] * (rows * cols)
  for line in by_line.values():
    line_txt = _clean_text(" ".join(line["tokens"]))
    if not line_txt:
      continue
    xs = sorted([c[0] for c in line["centers"]])
    ys = sorted([c[1] for c in line["centers"]])
    cx = xs[len(xs)//2]
    cy = ys[len(ys)//2]
    r = min(rows - 1, max(0, cy // cell_h))
    c = min(cols - 1, max(0, cx // cell_w))
    idx = r * cols + c
    if texts[idx]:
      texts[idx] = texts[idx] + " " + line_txt
    else:
      texts[idx] = line_txt

  # final cleanup pass per-cell
  for i, t in enumerate(texts):
    t = _clean_text(t)
    if len(t) <= 1:
      t = ""
    texts[i] = t

  # return cell texts
  return texts

# --------------------------- Retrieval (Text + CLIP over compressed cells) ---------------------------

def build_text_embedder():
  # sentence transformer on CPU (fast and compact)
  model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
  return model

def build_clip():
  # openclip ViT-B/32 on CPU
  model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"
  )
  tokenizer = open_clip.get_tokenizer("ViT-B-32")
  return model, preprocess, tokenizer

def compute_clip_image_emb_from_boxes(image_rgb: np.ndarray, boxes: np.ndarray, clip_model, clip_preprocess) -> np.ndarray:
  # crop each cell box, preprocess, encode with CLIP image encoder
  imgs = []
  for (y0, y1, x0, x1) in boxes:
    crop = image_rgb[y0:y1, x0:x1]
    pil = Image.fromarray(crop)
    imgs.append(clip_preprocess(pil).unsqueeze(0))
  batch = torch.cat(imgs, dim=0)
  embs = []
  bs = 64
  with torch.no_grad():
    for i in range(0, batch.size(0), bs):
      out = clip_model.encode_image(batch[i:i+bs])
      out = F.normalize(out, dim=-1)
      embs.append(out.cpu())
  return torch.cat(embs, dim=0).numpy()

# --------------------------- Visualization ---------------------------

def draw_topk_boxes(image_rgb: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                    top_indices: np.ndarray, outfile: str = "vis_top3.png"):
  # draw rectangles for top-k hits with their hybrid scores
  img = image_rgb[:, :, ::-1].copy()
  for rank, idx in enumerate(top_indices, 1):
    y0, y1, x0, x1 = boxes[idx]
    color = (0, 255 - 60 * (rank - 1), 0)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    label = f"#{rank} {scores[idx]:.3f}"
    cv2.putText(img, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
  cv2.imwrite(outfile, img)
  print(f"[OK] Saved top-k overlay -> {os.path.abspath(outfile)}")

def draw_score_heatmap_rc(image_rgb: np.ndarray, scores: np.ndarray, rows: int, cols: int,
                          outfile: str = "vis_heatmap.png", alpha: float = 0.45):
  # render a heatmap from a rows×cols score grid onto the page
  H, W = image_rgb.shape[:2]
  s = scores.copy()
  s -= s.min()
  if s.max() > 0:
    s /= s.max()
  heat = s.reshape(rows, cols)
  heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC)
  heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
  page = image_rgb[:, :, ::-1].copy()
  overlay = cv2.addWeighted(page, 1.0, heat_color, alpha, 0.0)
  cv2.imwrite(outfile, overlay)
  print(f"[OK] Saved heatmap overlay -> {os.path.abspath(outfile)}")

# --------------------------- Pretty Print + Answer Files -------------------------

def truncate(s: str, n: int = 140) -> str:
  # truncate helper for readable tables
  s = s.strip()
  return (s[:n] + "…") if len(s) > n else s

def print_topk_table(order, boxes, score_hybrid, score_text, score_clip, texts, rows, cols,
                     out_json="results_topk.json", out_md="results_topk.md"):
  # build rows for outputs
  table_rows = []
  for rank, idx in enumerate(order, 1):
    r, c = divmod(int(idx), cols)
    table_rows.append({
      "rank": rank,
      "idx": int(idx),
      "row": int(r),
      "col": int(c),
      "box": [int(x) for x in boxes[idx].tolist()],
      "score_hybrid": float(score_hybrid[idx]),
      "score_text": float(score_text[idx]),
      "score_clip": float(score_clip[idx]),
      "snippet": truncate(texts[idx] or "[no text]", 220),
    })

  # console table
  print("\nTop-3 results (ranked):")
  print("-" * 96)
  print(f"{'Rank':<6} {'Hybrid':<10} {'Text':<10} {'CLIP':<10} {'Cell(r,c)':<12} {'Snippet'}")
  print("-" * 96)
  for r in table_rows:
    print(f"{r['rank']:<6} {r['score_hybrid']:<10.3f} {r['score_text']:<10.3f} {r['score_clip']:<10.3f} "
          f"({r['row']},{r['col']})   {truncate(r['snippet'], 80)}")
  print("-" * 96)

  # write json
  with open(out_json, "w", encoding="utf-8") as f:
    json.dump(table_rows, f, ensure_ascii=False, indent=2)

  # write markdown
  with open(out_md, "w", encoding="utf-8") as f:
    f.write("| Rank | Hybrid | Text | CLIP | Cell (r,c) | Snippet |\n")
    f.write("|---:|---:|---:|---:|:---:|---|\n")
    for r in table_rows:
      f.write(f"| {r['rank']} | {r['score_hybrid']:.3f} | {r['score_text']:.3f} | {r['score_clip']:.3f} "
              f"| ({r['row']},{r['col']}) | {r['snippet'].replace('|','\\|')} |\n")

  # artifacts saved
  print(f"[OK] Saved Top-3 -> {os.path.abspath(out_json)}, {os.path.abspath(out_md)}")

def write_answer_summary(order, score_hybrid, score_text, score_clip, texts, boxes, rows, cols, out_path="answer.md"):
  # write a short markdown answer summary for the top hit
  best = int(order[0])
  r, c = divmod(best, cols)
  hy, tx, cl = score_hybrid[best], score_text[best], score_clip[best]
  y0, y1, x0, x1 = boxes[best].tolist()
  with open(out_path, "w", encoding="utf-8") as f:
    f.write("# Answer Summary\n\n")
    f.write(f"**Top match (cell {r},{c})**\n\n")
    f.write(f"- Hybrid score: **{hy:.3f}**  \n- Text score: **{tx:.3f}**  \n- CLIP score: **{cl:.3f}**\n\n")
    f.write(f"- Box: `y0={y0}, y1={y1}, x0={x0}, x1={x1}`\n\n")
    f.write("**Snippet:**\n\n")
    f.write("> " + (texts[best] or "[no text]") + "\n")
  print(f"[OK] Saved Answer Summary -> {os.path.abspath(out_path)}")

# --------------------------- Ollama Agent (Step 10b) ----------------------------

def build_agent_context(denoised_cells, order, score_hybrid, score_text, score_clip, texts, rows, cols, topn_cells=60):
  # rank cells by hybrid score and keep top-N for context
  ranked = [(int(i), float(score_hybrid[i])) for i in range(len(texts))]
  ranked.sort(key=lambda x: -x[1])
  keep = set([idx for idx, _ in ranked[:max(1, topn_cells)]])
  kept_cells = [c for c in denoised_cells if int(c["idx"]) in keep]
  kept_cells = kept_cells[:topn_cells]

  # denoised table with digit-aware truncation (numbers preserved)
  lines = ["# Denoised Cells (subset)", "", "| idx | r | c | text |", "|---:|---:|---:|---|"]
  for c in kept_cells:
    txt = c['text']
    if not re.search(r"\d", txt):
      txt = truncate(txt, 160)
    lines.append(f"| {c['idx']} | {c['row']} | {c['col']} | {txt.replace('|','\\|')} |")

  # retrieval top-k table
  top_table = ["", "# Retrieval Top-k", "", "| Rank | Hybrid | Text | CLIP | Cell (r,c) | Snippet |", "|---:|---:|---:|---:|:---:|---|"]
  for rank, idx in enumerate(order, 1):
    r, c = divmod(int(idx), cols)
    snip = texts[idx] or ""
    if not re.search(r"\d", snip):
      snip = truncate(snip, 180)
    top_table.append(f"| {rank} | {score_hybrid[idx]:.3f} | {score_text[idx]:.3f} | {score_clip[idx]:.3f} | ({r},{c}) | {snip.replace('|','\\|')} |")

  # return combined markdown
  return "\n".join(lines + top_table)

def call_ollama(url, model, system_prompt, user_prompt, num_ctx=4096, num_predict=512, temperature=0.2):
  # call Ollama HTTP chat endpoint
  if requests is None:
    return None, "The 'requests' package is not installed. Please 'pip install requests'."
  try:
    payload = {
      "model": model,
      "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
      ],
      "stream": False,
      "options": {
        "num_ctx": int(num_ctx),
        "temperature": float(temperature),
        "num_predict": int(num_predict)
      }
    }
    resp = requests.post(f"{url.rstrip('/')}/api/chat", json=payload, timeout=600)
    if resp.status_code != 200:
      return None, f"HTTP {resp.status_code}: {resp.text[:300]}"
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return content, None
  except Exception as e:
    return None, f"Ollama call failed: {e}"

# --------------------------- Main Orchestration ---------------------------

def main():
  # parse args and freeze grads
  args = parse_args()
  torch.set_grad_enabled(False)
  os.environ["TOKENIZERS_PARALLELISM"] = "false"

  # step 1: pdf -> image
  print("[Step 1] Rasterizing PDF...")
  t0 = time.perf_counter()
  img_rgb = pdf_to_image(args.pdf, args.dpi)
  H, W = img_rgb.shape[:2]
  print(f"[Info] Page: {H}x{W} @ {args.dpi} DPI in {time.perf_counter()-t0:.2f}s")

  # step 2: window-attention tokenizer (Swin) -> 256-D tokens at stride~16
  print("[Step 2] Window-attention tokenizer (Swin-Tiny) -> 256D tokens @ stride~16 ...")
  t1 = time.perf_counter()
  tokens_256, (h16, w16), (Hp, Wp), scale = swin_tokens_from_image(
    img_rgb, target_size=args.tokenizer_max_side, proj_dim=args.feature_dim, device="cpu"
  )
  N16 = h16 * w16
  print(f"[Info] Swin tokens: grid {h16}x{w16} = {N16} tokens (256-D) in {time.perf_counter()-t1:.2f}s")

  # step 3: 16× token compressor (sequence), /4 in H and /4 in W
  print("[Step 3] 16x token compressor (sequence via AvgPool2d /4 x /4) ...")
  t2 = time.perf_counter()
  comp_tokens, (hc, wc) = compress_tokens_2d(tokens_256, (h16, w16), pool_kernel=4)
  Nc = hc * wc
  print(f"[Info] Compressed sequence: grid {hc}x{wc} = {Nc} tokens (256-D) in {time.perf_counter()-t2:.2f}s")

  # step 4: global attention over compressed tokens @256-D
  print("[Step 4] Global attention over compressed tokens (256-D) ...")
  t3 = time.perf_counter()
  global_attn = SimpleSelfAttention(dim=256).eval()
  with torch.no_grad():
    tokens_ctx_256 = global_attn(comp_tokens)     # (Nc, 256)
  print(f"[Info] Context-mixed tokens: {tuple(tokens_ctx_256.shape)} in {time.perf_counter()-t3:.2f}s")

  # step 5: save compact tokens and build boxes for rows×cols grid
  print("[Step 5] Save compact tokens + build rows×cols boxes ...")
  boxes_rc = make_grid_boxes_rc(H, W, hc, wc)
  save_tokens(tokens_ctx_256.numpy(), boxes_rc, hc, wc, H, W, args.dpi, out="page_tokens_compact.npz")

  # step 6: OCR page -> map text to rows×cols cells
  print("[Step 6] OCR page -> map to rows×cols cells ...")
  t4 = time.perf_counter()
  texts = ocr_page_to_patch_texts_rc(img_rgb, hc, wc, confidence=args.confidence_score, lang=args.ocr_lang)
  with open("patch_texts.json", "w", encoding="utf-8") as f:
    json.dump({"grid": [hc, wc], "texts": texts}, f, ensure_ascii=False, indent=2)
  non_empty = [
    {"idx": int(i), "row": int(i // wc), "col": int(i % wc), "box": [int(x) for x in boxes_rc[i]], "text": texts[i]}
    for i in range(len(texts)) if texts[i]
  ]
  with open("patch_texts_denoised.json", "w", encoding="utf-8") as f:
    json.dump({"grid": [hc, wc], "cells": non_empty}, f, ensure_ascii=False, indent=2)
  print(f"[Info] OCR mapped in {time.perf_counter()-t4:.2f}s -> patch_texts.json, patch_texts_denoised.json")

  # step 7a: text embedder + scoring over Nc cells
  print("[Step 7a] Text embedder + scoring all cells ...")
  t5 = time.perf_counter()
  txt_model = build_text_embedder()
  patch_texts = [t if t else "[EMPTY]" for t in texts]
  patch_text_emb = txt_model.encode(patch_texts, normalize_embeddings=True)   # (Nc, d)
  q_txt = txt_model.encode([args.query], normalize_embeddings=True)[0]        # (d,)
  sims_text_all = patch_text_emb @ q_txt                                      # (Nc,)
  print(f"[Info] Text similarity computed in {time.perf_counter()-t5:.2f}s")

  # step 7b: CLIP vision scoring over the compressed cells (skipped if beta=0 or --no-clip)
  print("[Step 7b] CLIP vision scoring over compressed cells ...]")
  use_clip = not (args.no_clip or args.beta == 0.0)
  sims_clip_all = np.zeros_like(sims_text_all)
  if use_clip:
    t6 = time.perf_counter()
    clip_model, clip_preprocess, clip_tokenizer = build_clip()
    clip_img_emb = compute_clip_image_emb_from_boxes(img_rgb, boxes_rc, clip_model, clip_preprocess)
    qt = clip_tokenizer([args.query])
    with torch.no_grad():
      q_clip = clip_model.encode_text(qt)
      q_clip = F.normalize(q_clip, dim=-1)[0].cpu().numpy()
    sims_clip_all = clip_img_emb @ q_clip
    print(f"[Info] CLIP scoring done in {time.perf_counter()-t6:.2f}s")
  else:
    print("[Info] CLIP disabled (beta=0.0 or --no-clip)")

  # step 7c: hybrid scoring + top-k
  print("[Step 7c] Hybrid scoring + Top-K selection ...")
  alpha = args.alpha
  beta = args.beta if use_clip else 0.0
  score_hybrid = alpha * sims_text_all + beta * sims_clip_all
  order = np.argsort(-score_hybrid)[:args.topk]
  score_text = sims_text_all
  score_clip = sims_clip_all
  print(f"[Info] Top-{args.topk} selected. Best idx={int(order[0])} hybrid={float(score_hybrid[order[0]]):.3f}")

  # step 7d: results tables
  print("[Step 7d] Writing results tables ...")
  json_path = f"{args.out_prefix}.json"
  md_path = f"{args.out_prefix}.md"
  print_topk_table(order, boxes_rc, score_hybrid, score_text, score_clip, texts, hc, wc,
                   out_json=json_path, out_md=md_path)
  write_answer_summary(order, score_hybrid, score_text, score_clip, texts, boxes_rc, hc, wc,
                       out_path="answer.md")

  # step 8: visualizations
  print("[Step 8] Visualizations ...")
  draw_topk_boxes(img_rgb, boxes_rc, score_hybrid, order, outfile="vis_top3.png")
  draw_score_heatmap_rc(img_rgb, score_hybrid, hc, wc, outfile="vis_heatmap.png", alpha=0.45)

  # step 10b: optional ollama synthesis
  if args.ollama:
    print("[Step 9] Synthesizing a human-friendly answer with Ollama ...")
    try:
      with open("patch_texts_denoised.json", "r", encoding="utf-8") as f:
        den = json.load(f)
      den_cells = den.get("cells", [])
    except Exception as e:
      den_cells = []
      print(f"[Warn] Could not read patch_texts_denoised.json: {e}")

    ctx_md = build_agent_context(den_cells, order, score_hybrid, score_text, score_clip, texts, hc, wc,
                                 topn_cells=args.ollama_topn_cells)
    sys_prompt = (
      "You are an analytical assistant. You receive:\n"
      "1) A denoised table of OCR cells from a single PDF page (subset).\n"
      "2) A retrieval Top-k table with scores and snippets.\n"
      "Task: Read them, then answer the user query precisely and concisely, citing concrete values if present.\n"
      "Write a short executive summary first, then a numbered bullet explanation.\n"
      "If data is insufficient, say so and explain what is missing.\n"
      "Strictly use markdown for human friendly read and ease of understanding.\n"
      "When you cite values, quote the exact numbers from the tables. Put every numeric value in backticks and avoid rounding."
    )
    user_prompt = textwrap.dedent(f"""
    ## User Query
    {args.query}

    ## Context
    {ctx_md}
    """)

    if requests is None:
      answer, err = None, "Missing 'requests' package. Run: pip install requests"
    else:
      answer, err = call_ollama(
        url=args.ollama_url,
        model=args.ollama_model,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        num_ctx=args.ollama_num_ctx,
        num_predict=args.ollama_num_predict,
        temperature=args.ollama_temperature
      )

    if err:
      with open("agent_answer.md", "w", encoding="utf-8") as f:
        f.write("# Agent Answer\n\n")
        f.write("Error calling Ollama:\n\n")
        f.write(f"> {err}\n")
      print(f"[ERR] Ollama synthesis failed -> {err}")
      print(f"[OK] Wrote placeholder -> {os.path.abspath('agent_answer.md')}")
    else:
      with open("agent_answer.md", "w", encoding="utf-8") as f:
        f.write("# Agent Answer\n\n")
        f.write(answer.strip() + "\n")
      print(f"[OK] Saved agent answer -> {os.path.abspath('agent_answer.md')}")
  else:
    print("[Step 9] Ollama synthesis skipped (use --ollama to enable)")

  # final artifacts listing
  print("\nDone. Files:")
  print(f"- {os.path.abspath('page_tokens_compact.npz')}")
  print(f"- {os.path.abspath('patch_texts.json')}")
  print(f"- {os.path.abspath('patch_texts_denoised.json')}")
  print(f"- {os.path.abspath(json_path)}")
  print(f"- {os.path.abspath(md_path)}")
  print(f"- {os.path.abspath('answer.md')}")
  print(f"- {os.path.abspath('vis_top3.png')}")
  print(f"- {os.path.abspath('vis_heatmap.png')}")
  if args.ollama:
    print(f"- {os.path.abspath('agent_answer.md')}")

if __name__ == "__main__":
  main()

