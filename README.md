# DeepSeek-OCR Local Mimic (CPU-only, 1-page)
`original paper: (here)[https://arxiv.org/pdf/2510.18234]`
This demo reproduces the key **ideas** from DeepSeek-OCR in a CPU-friendly pipeline:

- **Layout-aware tokens**: 16×16 macro-grid subdivided 4×4 → **64×64 = 4096 micro-patches** (neighbor-aware via halo)
- **Widen → Attend → Compress**: 256-D tokens → **1024-D widen** → **full self-attention over 4096 tokens** → back to 256-D
- **Retrieval**: dual track
  - Text: OCR (Tesseract) + MiniLM embeddings → cosine
  - Vision: OpenCLIP (ViT-B/32) image/text similarity
  - **Hybrid score** = α·text + β·vision (default α=0.7, β=0.3)
- **Visualization**: top-k boxes and a heatmap overlay

> This is **inference-only** (no training) to demonstrate the **architecture** and why long-context + compression is powerful.

## Requirements
- Python 3.10+ recommended
- System package: Tesseract OCR
  - Ubuntu: `sudo apt-get install tesseract-ocr`
  - macOS (brew): `brew install tesseract`

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Generate pdf
```bash
python3 generate_pdf.py
```

## Run (Normal no AI Agent)
```bash
# use `--beta 0.3` when it works fine as i got some issues
python3 creditizens-ocr.py --pdf manga_kissa_report.pdf --query "evening attendance and snack sales" --beta 0.0 --out-prefix results_topk
# using prefiltering so should be fast when using `CLIP` (vision) `--beta 0.3`
python3 creditizens-ocr.py --pdf manga_kissa_report.pdf --query "evening attendance and snack sales" --beta 0.3 --clip-candidates 200 --out-prefix results_topk
```
**Had some issue so added some parameters:**
If this runs, the crash was in `OpenCLIP` weight download/mmap. After caches are clean, run again with `CLIP` (`--beta 0.3`)
`SIGBUS` is typically a truncated model file—clear `HF`/`CLIP` caches and pin the versions above.
Test text-only (`--beta 0.0`) to confirm the rest works; then re-enable `CLIP`.
Use the environment and code guards provided to stabilize CPU runs.


# torch special to get the `CPU` only stuff
```bash
# recommended fresh install
pip uninstall -y torch torchvision torchaudio

# CPU-only channel from PyTorch
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1

# Then the rest
pip install timm==0.9.16 sentence-transformers==2.7.0 open_clip_torch==2.24.0 \
            numpy==1.26.4 pillow==10.4.0 opencv-python==4.10.0.84 \
            pymupdf==1.24.9 pytesseract==0.3.13 reportlab==4.2.2 scikit-learn==1.5.2
```


# Recap
System dependency:
- Ubuntu/Debian → sudo apt-get install tesseract-ocr
- macOS → brew install tesseract

All other packages (ReportLab, PyMuPDF, etc.) are pure Python and installable via pip.

This setup supports:
- PDF generation (reportlab)
- PDF rasterization and parsing (pymupdf)
- OCR text extraction (pytesseract)
- Visual token encoding (torch, timm)
- Text embeddings (sentence-transformers)
- Vision-language similarity (open_clip_torch)
- Visualization and preprocessing (opencv-python, pillow, numpy)


## 💾 System Requirements

This project is `CPU-only`, no `GPU` or `CUDA` required.
However, it downloads several pretrained models (`MobileNetV3`, `SentenceTransformer`, `OpenCLIP`),
so plan for the following local resources:

| Generated data (tokens, heatmaps, etc.) | <100 MB per run | Temporary results          |
| Resource       | Minimum                    | Recommended | Notes                                                                                       |
| -------------- | -------------------------- | ----------- | ------------------------------------------------------------------------------------------- |
| **CPU**        | Dual-core                  | Quad-core   | Multithreading helps during image patch encoding.                                           |
| **RAM**        | 8 GB                       | 16 GB       | The 4096-token attention step briefly allocates a few GB of tensors.                        |
| **Disk space** | 6–8 GB free                | 10 GB+      | ~2 GB Hugging Face cache, ~2 GB OpenCLIP weights, ~1 GB PyTorch libs, rest temporary files. |
| **OS**         | Linux / macOS / WSL Ubuntu | —           | Tesseract OCR must be installed system-wide.                                                |
| **Python**     | 3.10 – 3.12                | —           | Virtual-env or `.venv` recommended.                                                         |


| Component                               | Approx. Size    | Purpose                    |
| --------------------------------------- | --------------- | -------------------------- |
| MobileNetV3-Small (timm)                | ~10 MB          | Image patch encoder        |
| Sentence-Transformer (MiniLM)           | ~400 MB         | Text embeddings            |
| OpenCLIP ViT-B/32                       | ~2.0 GB         | Vision–language retrieval  |
| Torch + deps                            | ~2.5 GB         | Core deep-learning runtime |
| Hugging Face / CLIP cache               | 1–2 GB          | Model weight storage       |
| Generated data (tokens, heatmaps, etc.) | <100 MB per run | Temporary results          |


# RUN SPECIAL WITH OLLAMA AI AGENTS
# 1) Install Ollama (Linux)
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
# 2) Start the server (new terminal or background)
```bash
ollama serve
```
# 3) Pull a small, sharp model (pick ONE of these)
```bash
ollama pull qwen2.5:1.5b-instruct
# or
ollama pull llama3.2:1b-instruct
# or
ollama pull phi3:mini
```
# for context length can change it with this:
```bash
--ollama-num-ctx
```
# 4) Ensure 'requests' is installed for the script
```bash
pip install requests
```
# 5) Run your pipeline + agent
```bash
python3 creditizens-ocr.py \
  --pdf manga_kissa_report.pdf \
  --query "evening attendance and snack sales" \
  --beta 0.3 \
  --ollama \
  --ollama-model qwen2.5:1.5b-instruct \
  --ollama-num-ctx 4096 \
  --ollama-num-predict 512 \
  --ollama-topn-cells 60 \
  --out-prefix results_topk
```

# **IMPORTANT**: for the run command in terminal check script files top comment as it might change a bit as have added parameters like `--condidence-score`
