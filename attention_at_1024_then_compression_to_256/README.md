# GPT Proposed Improved Sequence (But not following DeepSeek-OCR Paper Steps)
Still satisfactory even if not following step strictly like `DeepSeek-OCR Paper`:
- here attention done at 1024 Dimension instead of 256 Dimension before compression

1) PDF -> image
2) 16x16 macro-grid, each macro split 4x4 => 64x64 micro-grid = 4096 tokens (neighbor-aware halo)
3) Visual encoder (MobileNetV3-Small) -> 256-D tokens
4) Widen 256 -> 1024
5) Full self-attention over 4096 tokens at 1024-D (true long-context step)
6) Compress 1024 -> 256 (storable compact representation)
7) OCR page, map words to micro-cells
8) Dual retrieval (Text via MiniLM, Vision via OpenCLIP) + hybrid scoring
9) Top-k results + heatmap visualization
