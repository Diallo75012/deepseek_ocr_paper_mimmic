"""
demo.py — CPU-only, 1-page DeepSeek-OCR mimic for @creditizens

What it shows:
1) PDF -> image
2) 16x16 macro-grid => 256 tokens (layout-aligned)
3) Visual encoder (MobileNetV3-Small) -> 256-D tokens
4) Widen 256 -> 1024 (capacity for token-space expansion)
5) Stretch 16x16 tokens -> 64x64 = 4096 tokens (bilinear upsampling in token space, neighbor-aware)
6) Compress 1024 -> 256 (bottleneck before attention)
7) Full self-attention over 4096 tokens at 256-D (true long-context step, cheaper)
8) OCR page, map words to 64x64 micro-cells
9) Dual retrieval (Text via MiniLM, Vision via OpenCLIP@macro then upsample) + hybrid scoring
10) Top-k results + heatmap visualization
11) (Optional) Local Ollama agent to synthesize a friendly answer

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
  --pdf manga_kissa_report.pdf \
  --query "evening attendance and snack sales" \
  --beta 0.0 \
  --ollama \
  --ollama-model qwen2.5:1.5b-instruct \
  --ollama-num-ctx 4096 \
  --ollama-num-predict 512 \
  --ollama-topn-cells 60 \
  --out-prefix results_topk
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
  ap = argparse.ArgumentParser(description="DeepSeek-OCR mimic (CPU-only, 1-page)")
  ap.add_argument("--pdf", required=True, help="Input 1-page PDF path")
  ap.add_argument("--dpi", type=int, default=300, help="Raster DPI (300-400 is good)")
  ap.add_argument("--macro-grid", type=int, default=16, help="Base macro grid (16 like paper) -> 256 tokens")
  ap.add_argument("--encoder-size", type=int, default=160, help="CNN input size (smaller than 224 for CPU speed)")
  ap.add_argument("--feature-dim", type=int, default=256, help="Compact token dim")
  ap.add_argument("--widen-dim", type=int, default=1024, help="Expanded dim before upsampling")
  ap.add_argument("--ocr-lang", default="eng", help="Tesseract language (e.g., eng, eng+fra)")
  ap.add_argument("--query", required=True, help="User query for retrieval")
  ap.add_argument("--topk", type=int, default=3, help="Top-k results to show")
  ap.add_argument("--alpha", type=float, default=0.7, help="Weight for text score")
  ap.add_argument("--beta", type=float, default=0.3, help="Weight for CLIP vision score (set 0.0 to skip CLIP)")
  ap.add_argument("--no-clip", action="store_true", help="Force skip CLIP (debug)")
  ap.add_argument("--out-prefix", default="results_topk", help="Basename for result files")
  ap.add_argument("--clip-candidates", type=int, default=200, help="Run CLIP only on top-N text candidates (macro level)")
  ap.add_argument("--ollama", action="store_true", help="Use a local Ollama model to synthesize a final human answer")
  ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
  ap.add_argument("--ollama-model", default="qwen2.5:1.5b-instruct", help="Ollama model name (CPU-friendly)")
  ap.add_argument("--ollama-num-predict", type=int, default=512, help="Max tokens to generate")
  ap.add_argument("--ollama-num-ctx", type=int, default=4096, help="Context window size")
  ap.add_argument("--ollama-temperature", type=float, default=0.2, help="Generation temperature")
  ap.add_argument("--ollama-topn-cells", type=int, default=60, help="How many high-score cells to include in agent context")
  return ap.parse_args()

# --------------------------- Step 1: PDF -> Image ---------------------------

def pdf_to_image(pdf_path: str, dpi: int = 300) -> np.ndarray:
  doc = fitz.open(pdf_path)
  page = doc[0]
  pix = page.get_pixmap(dpi=dpi, alpha=False)
  img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
  return img

# --------------------------- Grid helpers ---------------------------

def macrogrid_patches(image_rgb: np.ndarray, macro: int, size: int):
  H, W = image_rgb.shape[:2]
  mh, mw = H // macro, W // macro
  patches, boxes = [], []
  for mr in range(macro):
    for mc in range(macro):
      y0 = mr * mh
      y1 = (mr + 1) * mh
      x0 = mc * mw
      x1 = (mc + 1) * mw
      crop = image_rgb[y0:y1, x0:x1]
      crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
      patches.append(crop)
      boxes.append((int(y0), int(y1), int(x0), int(x1)))
  patches = np.stack(patches, 0)
  boxes = np.array(boxes)
  return patches, boxes

def make_grid_boxes(H: int, W: int, side: int):
  cell_h, cell_w = H // side, W // side
  boxes = []
  for r in range(side):
    for c in range(side):
      y0 = r * cell_h
      y1 = (r + 1) * cell_h
      x0 = c * cell_w
      x1 = (c + 1) * cell_w
      boxes.append((y0, y1, x0, x1))
  return np.array(boxes, dtype=np.int32)

# --------------------------- Step 3: Visual Encoder -> 256-D tokens ---------------------------

def build_backbone_and_proj(feature_dim: int, device: str = "cpu"):
  backbone = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=0).eval().to(device)
  proj_to_256 = nn.Linear(backbone.num_features, feature_dim, bias=False).to(device)
  return backbone, proj_to_256

def encode_patches_to_256(patches_rgb: np.ndarray, backbone, proj_to_256, device: str = "cpu") -> torch.Tensor:
  t = torch.from_numpy(patches_rgb).permute(0, 3, 1, 2).float() / 255.0
  feats = []
  bs = 128
  with torch.no_grad():
    for i in range(0, t.size(0), bs):
      batch = t[i:i + bs].to(device)
      f = backbone(batch)
      f256 = proj_to_256(f)
      feats.append(f256.cpu())
  return torch.cat(feats, dim=0)

# --------------------------- Step 4: Widen 256 -> 1024 ---------------------------

def build_widen(feature_dim: int, widen_dim: int, device: str = "cpu"):
  return nn.Linear(feature_dim, widen_dim, bias=False).to(device)

# --------------------------- Step 5: Stretch 16x16 -> 64x64 in token space ---------------------------

def upsample_tokens_bilinear(tokens_2d_1024: torch.Tensor, in_side: int, out_side: int) -> torch.Tensor:
  # tokens_2d_1024: (N=side*side, 1024)
  # reshape to (1, C=1024, H=in_side, W=in_side)
  x = tokens_2d_1024.view(in_side, in_side, -1).permute(2, 0, 1).unsqueeze(0)
  x = F.interpolate(x, size=(out_side, out_side), mode="bilinear", align_corners=False)
  x = x.squeeze(0).permute(1, 2, 0).contiguous().view(out_side * out_side, -1)
  return x

# --------------------------- Step 6/7: Compress and Attention ---------------------------

class SimpleSelfAttention(nn.Module):
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

def save_tokens(tokens_256: np.ndarray, boxes: np.ndarray, grid_side: int,
                H: int, W: int, dpi: int, macro: int, out="page_tokens_4096.npz"):
  np.savez(out,
           tokens=tokens_256.astype(np.float32),
           boxes=boxes,
           grid=np.array([grid_side, grid_side], dtype=np.int32),
           meta=np.array([H, W, dpi], dtype=np.int32),
           macro=np.array([macro], dtype=np.int32))
  print(f"[OK] Saved tokens -> {os.path.abspath(out)}")

# --------------------------- Step 8: OCR page & map words to micro-cells ---------------------------

def _clean_text(s: str) -> str:
  s = re.sub(r"\s+", " ", s)
  s = s.strip(" -:;,.")
  return s.strip()

def ocr_page_to_patch_texts(img_rgb: np.ndarray, grid_side: int, lang: str = "eng") -> list[str]:
  H, W = img_rgb.shape[:2]
  cell_h, cell_w = H // grid_side, W // grid_side
  pil = Image.fromarray(img_rgb)
  ocr = pytesseract.image_to_data(
    pil, lang=lang, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT
  )
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
    if conf < 60:
      continue
    line_id = (ocr["block_num"][i], ocr["par_num"][i], ocr["line_num"][i])
    x, y, w, h = ocr["left"][i], ocr["top"][i], ocr["width"][i], ocr["height"][i]
    cx, cy = x + w // 2, y + h // 2
    if line_id not in by_line:
      by_line[line_id] = {"tokens": [], "centers": []}
    by_line[line_id]["tokens"].append(txt)
    by_line[line_id]["centers"].append((cx, cy))
  texts = [""] * (grid_side * grid_side)
  for line in by_line.values():
    line_txt = _clean_text(" ".join(line["tokens"]))
    if not line_txt:
      continue
    xs = sorted([c[0] for c in line["centers"]])
    ys = sorted([c[1] for c in line["centers"]])
    cx = xs[len(xs)//2]
    cy = ys[len(ys)//2]
    r = min(grid_side - 1, max(0, cy // cell_h))
    c = min(grid_side - 1, max(0, cx // cell_w))
    idx = r * grid_side + c
    if texts[idx]:
      texts[idx] = texts[idx] + " " + line_txt
    else:
      texts[idx] = line_txt
  for i, t in enumerate(texts):
    t = _clean_text(t)
    if len(t) <= 1:
      t = ""
    texts[i] = t
  return texts

# --------------------------- Step 9: Dual Retrieval (Text + CLIP@macro -> upsample) ---------------------------

def build_text_embedder():
  model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
  return model

def build_clip():
  model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"
  )
  tokenizer = open_clip.get_tokenizer("ViT-B-32")
  return model, preprocess, tokenizer

def compute_clip_image_emb(patches_rgb: np.ndarray, clip_model, clip_preprocess) -> np.ndarray:
  imgs = []
  for p in patches_rgb:
    imgs.append(clip_preprocess(Image.fromarray(p)).unsqueeze(0))
  batch = torch.cat(imgs, dim=0)
  embs = []
  bs = 64
  with torch.no_grad():
    for i in range(0, batch.size(0), bs):
      out = clip_model.encode_image(batch[i:i+bs])
      out = F.normalize(out, dim=-1)
      embs.append(out.cpu())
  return torch.cat(embs, dim=0).numpy()

# --------------------------- Step 10: Visualization ---------------------------

def draw_topk_boxes(image_rgb: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                    top_indices: np.ndarray, outfile: str = "vis_top3.png"):
  img = image_rgb[:, :, ::-1].copy()
  for rank, idx in enumerate(top_indices, 1):
    y0, y1, x0, x1 = boxes[idx]
    color = (0, 255 - 60 * (rank - 1), 0)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    label = f"#{rank} {scores[idx]:.3f}"
    cv2.putText(img, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
  cv2.imwrite(outfile, img)
  print(f"[OK] Saved top-k overlay -> {os.path.abspath(outfile)}")

def draw_score_heatmap(image_rgb: np.ndarray, scores: np.ndarray, grid_side: int,
                       outfile: str = "vis_heatmap.png", alpha: float = 0.45):
  H, W = image_rgb.shape[:2]
  s = scores.copy()
  s -= s.min()
  if s.max() > 0:
    s /= s.max()
  heat = s.reshape(grid_side, grid_side)
  heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC)
  heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
  page = image_rgb[:, :, ::-1].copy()
  overlay = cv2.addWeighted(page, 1.0, heat_color, alpha, 0.0)
  cv2.imwrite(outfile, overlay)
  print(f"[OK] Saved heatmap overlay -> {os.path.abspath(outfile)}")

# --------------------------- Pretty Print + Answer Files -------------------------

def truncate(s: str, n: int = 140) -> str:
  s = s.strip()
  return (s[:n] + "…") if len(s) > n else s

def print_topk_table(order, boxes, score_hybrid, score_text, score_clip, texts, grid_side,
                     out_json="results_topk.json", out_md="results_topk.md"):
  rows = []
  for rank, idx in enumerate(order, 1):
    r, c = divmod(int(idx), grid_side)
    rows.append({
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
  print("\nTop-3 results (ranked):")
  print("-" * 96)
  print(f"{'Rank':<6} {'Hybrid':<10} {'Text':<10} {'CLIP':<10} {'Cell(r,c)':<12} {'Snippet'}")
  print("-" * 96)
  for r in rows:
    print(f"{r['rank']:<6} {r['score_hybrid']:<10.3f} {r['score_text']:<10.3f} {r['score_clip']:<10.3f} "
          f"({r['row']},{r['col']})   {truncate(r['snippet'], 80)}")
  print("-" * 96)
  with open(out_json, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
  with open(out_md, "w", encoding="utf-8") as f:
    f.write("| Rank | Hybrid | Text | CLIP | Cell (r,c) | Snippet |\n")
    f.write("|---:|---:|---:|---:|:---:|---|\n")
    for r in rows:
      f.write(f"| {r['rank']} | {r['score_hybrid']:.3f} | {r['score_text']:.3f} | {r['score_clip']:.3f} "
              f"| ({r['row']},{r['col']}) | {r['snippet'].replace('|','\\|')} |\n")
  print(f"[OK] Saved Top-3 -> {os.path.abspath(out_json)}, {os.path.abspath(out_md)}")

def write_answer_summary(order, score_hybrid, score_text, score_clip, texts, boxes, grid_side, out_path="answer.md"):
  best = int(order[0])
  r, c = divmod(best, grid_side)
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

# --------------------------- Ollama Agent (Step 11) ----------------------------

def build_agent_context(denoised_cells, order, score_hybrid, score_text, score_clip, texts, grid_side, topn_cells=60):
  ranked = [(int(i), float(score_hybrid[i])) for i in range(len(texts))]
  ranked.sort(key=lambda x: -x[1])
  keep = set([idx for idx, _ in ranked[:max(1, topn_cells)]])
  kept_cells = [c for c in denoised_cells if int(c["idx"]) in keep]
  kept_cells = kept_cells[:topn_cells]
  lines = ["# Denoised Cells (subset)", "", "| idx | r | c | text |", "|---:|---:|---:|---|"]
  for c in kept_cells:
    lines.append(f"| {c['idx']} | {c['row']} | {c['col']} | {truncate(c['text'], 160).replace('|','\\|')} |")
  top_table = ["", "# Retrieval Top-k", "", "| Rank | Hybrid | Text | CLIP | Cell (r,c) | Snippet |", "|---:|---:|---:|---:|:---:|---|"]
  for rank, idx in enumerate(order, 1):
    r, c = divmod(int(idx), grid_side)
    top_table.append(f"| {rank} | {score_hybrid[idx]:.3f} | {score_text[idx]:.3f} | {score_clip[idx]:.3f} | ({r},{c}) | {truncate(texts[idx], 180).replace('|','\\|')} |")
  return "\n".join(lines + top_table)

def call_ollama(url, model, system_prompt, user_prompt, num_ctx=4096, num_predict=512, temperature=0.2):
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
  args = parse_args()
  torch.set_grad_enabled(False)
  os.environ["TOKENIZERS_PARALLELISM"] = "false"

  print("[Step 1] Rasterizing PDF...")
  t0 = time.perf_counter()
  img_rgb = pdf_to_image(args.pdf, args.dpi)
  H, W = img_rgb.shape[:2]
  print(f"[Info] Page: {H}x{W} @ {args.dpi} DPI in {time.perf_counter()-t0:.2f}s")

  print("[Step 2] Building 16x16 macro grid -> 256 tokens...")
  t1 = time.perf_counter()
  macro = args.macro_grid
  macro_patches, macro_boxes = macrogrid_patches(img_rgb, macro=macro, size=args.encoder_size)
  print(f"[Info] Macro grid: {macro}x{macro} = {macro_patches.shape[0]} patches in {time.perf_counter()-t1:.2f}s")

  print("[Step 3] Visual encoder -> 256D tokens (macro level)...")
  t2 = time.perf_counter()
  backbone, proj_to_256 = build_backbone_and_proj(args.feature_dim, device="cpu")
  macro_tokens_256 = encode_patches_to_256(macro_patches, backbone, proj_to_256, device="cpu")   # (256, 256)
  print(f"[Info] Encoded macro tokens: {tuple(macro_tokens_256.shape)} in {time.perf_counter()-t2:.2f}s")

  print("[Step 4] Widen 256 -> 1024 (macro level)...")
  t3 = time.perf_counter()
  widen = build_widen(args.feature_dim, args.widen_dim, device="cpu")
  macro_tokens_1024 = widen(macro_tokens_256).contiguous()                                       # (256, 1024)
  print(f"[Info] Widened macro tokens: {tuple(macro_tokens_1024.shape)} in {time.perf_counter()-t3:.2f}s")

  print("[Step 5] Stretch 16x16 -> 64x64 in token space (neighbor-aware bilinear)...")
  t4 = time.perf_counter()
  grid_in = macro
  grid_out = macro * 4
  tokens_1024_4096 = upsample_tokens_bilinear(macro_tokens_1024, in_side=grid_in, out_side=grid_out)  # (4096, 1024)
  micro_boxes = make_grid_boxes(H, W, grid_out)
  print(f"[Info] Token grid stretched: {grid_out}x{grid_out} = {tokens_1024_4096.shape[0]} in {time.perf_counter()-t4:.2f}s")

  print("[Step 6] Compress 1024 -> 256 (bottleneck before attention)...")
  t5 = time.perf_counter()
  proj_down = nn.Linear(args.widen_dim, args.feature_dim, bias=False)
  tokens_256_4096 = proj_down(tokens_1024_4096)                                                   # (4096, 256)
  print(f"[Info] Compressed tokens: {tuple(tokens_256_4096.shape)} in {time.perf_counter()-t5:.2f}s")

  print("[Step 7] 4096-token self-attention @ 256-D...")
  t6 = time.perf_counter()
  attn = SimpleSelfAttention(args.feature_dim).eval()
  tokens_ctx_256 = attn(tokens_256_4096)                                                         # (4096, 256)
  print(f"[Info] Context-mixed tokens: {tuple(tokens_ctx_256.shape)} in {time.perf_counter()-t6:.2f}s")

  print("[Step 7b] Save compact long-context tokens...")
  save_tokens(tokens_ctx_256.detach().cpu().numpy(), micro_boxes, grid_out, H, W, args.dpi, macro)

  print("[Step 8] OCR page -> map to 64x64 micro-cells...")
  t7 = time.perf_counter()
  texts = ocr_page_to_patch_texts(img_rgb, grid_out, lang=args.ocr_lang)
  with open("patch_texts.json", "w", encoding="utf-8") as f:
    json.dump({"grid": grid_out, "texts": texts}, f, ensure_ascii=False, indent=2)
  non_empty = [
    {"idx": int(i), "row": int(i // grid_out), "col": int(i % grid_out), "box": [int(x) for x in micro_boxes[i]], "text": texts[i]}
    for i in range(len(texts)) if texts[i]
  ]
  with open("patch_texts_denoised.json", "w", encoding="utf-8") as f:
    json.dump({"grid": grid_out, "cells": non_empty}, f, ensure_ascii=False, indent=2)
  print(f"[Info] OCR mapped in {time.perf_counter()-t7:.2f}s -> patch_texts.json, patch_texts_denoised.json")

  print("[Step 9a] Text embedder + scoring all 4096 cells...")
  t8 = time.perf_counter()
  txt_model = build_text_embedder()
  patch_texts = [t if t else "[EMPTY]" for t in texts]
  patch_text_emb = txt_model.encode(patch_texts, normalize_embeddings=True)                       # (4096, d)
  q_txt = txt_model.encode([args.query], normalize_embeddings=True)[0]                            # (d,)
  sims_text_all = patch_text_emb @ q_txt                                                          # (4096,)
  print(f"[Info] Text similarity computed in {time.perf_counter()-t8:.2f}s")

  print("[Step 9b] CLIP vision scoring (macro -> upsample) ...")
  use_clip = not (args.no_clip or args.beta == 0.0)
  sims_clip_all = np.zeros_like(sims_text_all)
  if use_clip:
    t9 = time.perf_counter()
    Ncand = min(args.clip_candidates, macro_tokens_256.shape[0])
    print(f"[Info] Using CLIP on macro patches (16x16={macro*macro}), prefilter top-{Ncand} by text over 4096 via macro binning")

    # map 4096 text scores back to 16x16 macro bins by averaging 4x4 regions
    text_map_64 = sims_text_all.reshape(grid_out, grid_out)
    text_map_16 = cv2.resize(text_map_64.astype(np.float32), (macro, macro), interpolation=cv2.INTER_AREA)
    cand_flat = np.argsort(-text_map_16.flatten())[:Ncand]

    print("[Info] Loading OpenCLIP weights (first time may take a bit)...")
    clip_model, clip_preprocess, clip_tokenizer = build_clip()

    print("[Info] Encoding candidate macro patches with CLIP image encoder...")
    cand_patches = macro_patches[cand_flat]
    cand_img_emb = compute_clip_image_emb(cand_patches, clip_model, clip_preprocess)

    print("[Info] Encoding query with CLIP text encoder...")
    qt = clip_tokenizer([args.query])
    with torch.no_grad():
      q_clip = clip_model.encode_text(qt)
      q_clip = F.normalize(q_clip, dim=-1)[0].cpu().numpy()

    print("[Info] Scoring CLIP cosine similarities for candidates...")
    sims_clip_macro = np.zeros(macro * macro, dtype=np.float32)
    sims_clip_macro[cand_flat] = (cand_img_emb @ q_clip).astype(np.float32)

    # upsample macro CLIP map to 64x64 to match 4096 cells
    clip_map_16 = sims_clip_macro.reshape(macro, macro)
    clip_map_64 = cv2.resize(clip_map_16, (grid_out, grid_out), interpolation=cv2.INTER_CUBIC)
    sims_clip_all = clip_map_64.reshape(grid_out * grid_out)
    print(f"[Info] CLIP macro->micro map computed in {time.perf_counter()-t9:.2f}s")
  else:
    print("[Info] CLIP disabled (beta=0.0 or --no-clip)")

  print("[Step 9c] Hybrid scoring + Top-K selection...")
  alpha = args.alpha
  beta = args.beta if use_clip else 0.0
  score_hybrid = alpha * sims_text_all + beta * sims_clip_all
  order = np.argsort(-score_hybrid)[:args.topk]
  score_text = sims_text_all
  score_clip = sims_clip_all
  print(f"[Info] Top-{args.topk} selected. Best idx={int(order[0])} hybrid={float(score_hybrid[order[0]]):.3f}")

  print("[Step 9d] Writing results tables...")
  json_path = f"{args.out_prefix}.json"
  md_path = f"{args.out_prefix}.md"
  print_topk_table(order, micro_boxes, score_hybrid, score_text, score_clip, texts, grid_out,
                   out_json=json_path, out_md=md_path)
  write_answer_summary(order, score_hybrid, score_text, score_clip, texts, micro_boxes, grid_out,
                       out_path="answer.md")

  print("[Step 10] Visualizations...")
  draw_topk_boxes(img_rgb, micro_boxes, score_hybrid, order, outfile="vis_top3.png")
  draw_score_heatmap(img_rgb, score_hybrid, grid_out, outfile="vis_heatmap.png", alpha=0.45)

  if args.ollama:
    print("[Step 11] Synthesizing a human-friendly answer with Ollama...")
    try:
      with open("patch_texts_denoised.json", "r", encoding="utf-8") as f:
        den = json.load(f)
      den_cells = den.get("cells", [])
    except Exception as e:
      den_cells = []
      print(f"[Warn] Could not read patch_texts_denoised.json: {e}")

    ctx_md = build_agent_context(den_cells, order, score_hybrid, score_text, score_clip, texts, grid_out, topn_cells=args.ollama_topn_cells)
    sys_prompt = (
      "You are an analytical assistant. You receive:\n"
      "1) A denoised table of OCR cells from a single PDF page (subset).\n"
      "2) A retrieval Top-k table with scores and snippets.\n"
      "Task: Read them, then answer the user query precisely and concisely, citing concrete values if present.\n"
      "Write a short executive summary first, then a numbered bullet explanation.\n"
      "If data is insufficient, say so and explain what is missing.\n"
      "Strictly use markdown for human friendly read and ease of understanding."
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
    print("[Step 11] Ollama synthesis skipped (use --ollama to enable)")

  print("\nDone. Files:")
  print(f"- {os.path.abspath('page_tokens_4096.npz')}")
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
