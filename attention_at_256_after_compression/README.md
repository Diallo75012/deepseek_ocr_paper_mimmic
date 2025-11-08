# DeepSeek-OCR Exact Steps
Here we follow the exact steps by doing `ttention` after compression

1) 16×16 grid → 256 tokens: coarse layout tokens aligned with document structure (cheap).
2) CNN → 256-D: visual encoding into compact semantic features.
3) Widen to 1024-D: richer capacity for token-space interpolation/expansion.
4) Stretch to 4096 tokens (in token space): encoding step that increases sequence length with neighbor-aware bilinear upsampling over the 16×16 token grid,
   producing a 64×64 token field (not image patches).
5) Compress to 256-D before attention: reduce per-token width so attention over 4096 tokens is cheaper.
6) Attention at 256-D: long-context mixing across the whole page with controlled compute.
7) Retrieve
8) Use AI Agent for answer to Human
