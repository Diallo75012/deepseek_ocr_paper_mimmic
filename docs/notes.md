# Creditizens – DeepSeek-OCR Local Demo (Manga Kissa Shibuya)

Hi everyone, welcome back to Creditizens!

Today I’m showing how we can locally reproduce, on a simple laptop, the core logic of the **DeepSeek-OCR** paper — but using a friendly example: the “Manga Kissa Shibuya” daily report.

---

## Scene 1 – Concept
DeepSeek-OCR was revolutionary because it doesn’t just read text.
It looks at **the whole document layout**, cuts it into grid patches, mixes them with **neighbor awareness**, expands the representation to a higher dimension, runs **long-context attention (4096 tokens)**, and then compresses back to a smaller size.

This lets the system understand text, tables, and figures together — maintaining logical flow and spatial relations.

---

## Scene 2 – Our local mimic
We’ll run everything CPU-only:

1. Rasterize the PDF page (the Manga Kissa report).
2. Build a 16×16 macro-grid subdivided 4×4 → 64×64 micro-grid (4096 tokens total).
3. Encode each micro-patch with **MobileNetV3**, project to 256 dimensions.
4. Expand to 1024 dimensions (widen).
5. Apply full self-attention across all 4096 tokens.
6. Compress back to 256 dimensions (compact representation).
7. Extract text with **Tesseract OCR**.
8. Retrieve relevant regions using a **hybrid score**:
   - Text cosine similarity (MiniLM)
   - Visual similarity (OpenCLIP ViT-B/32)
   - Combined hybrid score = 0.7 · text + 0.3 · vision
9. Display top-3 results and a heatmap overlay.

---

## Scene 3 – Run (Normally without AI Agents)
So it is easy we have set `flags` and we activate `ollama` agent if needed but need to change the model accordingly in the script
```bash
# use `--beta 0.3` when it works fine as i got some issues
python3 creditizens-ocr.py --pdf manga_kissa_report.pdf --query "evening attendance and snack sales" --beta 0.0 --out-prefix results_topk
# using prefiltering so should be fast when using `CLIP` (vision) `--beta 0.3`
python3 creditizens-ocr.py --pdf manga_kissa_report.pdf --query "evening attendance and snack sales" --beta 0.3 --clip-candidates 200 --out-prefix results_topk
```

## 3" - Run (With AI Agents)
### 1) Install Ollama (Linux)
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
### 2) Start the server (new terminal or background)
```bash
ollama serve
```
### 3) Pull a small, sharp model (pick ONE of these)
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
### 4) Ensure 'requests' is installed for the script
```bash
pip install requests
```
### 5) Run your pipeline + agent
so here we use extra flags with the model that we want but has to be installed already and available in `ollama`
```bash
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
```

**Had some issue so added some parameters:**
If this runs, the crash was in `OpenCLIP` weight download/mmap. After caches are clean, run again with `CLIP` (`--beta 0.3`)
`SIGBUS` is typically a truncated model file—clear `HF`/`CLIP` caches and pin the versions above.
Test text-only (`--beta 0.0`) to confirm the rest works; then re-enable `CLIP`.
Use the environment and code guards provided to stabilize CPU runs.

## Explain **Outputs:**
- vis_top3.png → shows three highlighted boxes with scores.
- vis_heatmap.png → red areas indicate highest relevance.


________________________________________________________
```bash
widen → attend → compress, enabling longer document understanding on small models.
Encourage viewers to read the original DeepSeek-OCR paper for technical depth
and to imagine how these ideas might extend future RAG systems.
```


# Issues
- **if issues try to cehck which import is not working properly by running this in terminal**
```bash
python - <<'PY'
import torch; print("torch ok")
import timm; print("timm ok")
import sentence_transformers; print("sbert ok")
import open_clip; print("open_clip ok")
PY
```
