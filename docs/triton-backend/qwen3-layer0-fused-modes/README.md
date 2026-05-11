# Qwen3 Layer0 Fused-Kernel Mode Graphs

This directory contains PE=16 layer0 HF-to-nncase graphs generated with the same
layout and visual conventions as `docs/triton-backend/qwen3-compute-layer0.*`.
Both graphs use `main_segment_1_prim`, where ordinal 04 is the all-to-all
`SBP=(B,S(0)) -> SBP=(S(0),B)` reshard record.

- [`--fused-kernel=off`](./qwen3-off-layer0.md)
  - graph: [`qwen3-off-layer0.svg`](./qwen3-off-layer0.svg)
- [`--fused-kernel=compute-ccl`](./qwen3-compute-ccl-layer0.md)
  - graph: [`qwen3-compute-ccl-layer0.svg`](./qwen3-compute-ccl-layer0.svg)

Regenerate:

```bash
python docs/triton-backend/qwen3-layer0-fused-modes/generate_qwen3_layer0_fused_modes.py
```
