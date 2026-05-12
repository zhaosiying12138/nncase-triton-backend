# Qwen3 Layer0 Fused-Kernel Mode Graphs

This directory contains PE=16 layer0 HF-to-nncase graphs generated with the same
layout and visual conventions as `docs/triton-backend/qwen3-compute-layer0.*`.
The off and compute-ccl graphs use `main_segment_1_prim`, where ordinal 04
is the all-to-all `SBP=(B,S(0)) -> SBP=(S(0),B)` reshard record.

`compute-ccl` is CCL-only in this directory: it does not contain
`fusion.cuda.*` compute launches, and opportunistic GRS fusion is shown as
`ccl_tail.<producer>.grs` metadata plus an elided explicit GRS ordinal.

- [`--fused-kernel=off`](./qwen3-off-layer0.md)
  - graph: [`qwen3-off-layer0.svg`](./qwen3-off-layer0.svg)
- [`--fused-kernel=compute-ccl`](./qwen3-compute-ccl-layer0.md)
  - graph: [`qwen3-compute-ccl-layer0.svg`](./qwen3-compute-ccl-layer0.svg)
- [`vLLM TP=16 vs nncase --fused-kernel=off`](./qwen3-vllm-tp16-vs-nncase-off-layer0.md)
  - graph: [`qwen3-vllm-tp16-vs-nncase-off-layer0.svg`](./qwen3-vllm-tp16-vs-nncase-off-layer0.svg)
- [`off` vs `compute-ccl` CCL-tail comparison](./qwen3-off-vs-compute-ccl-layer0.md)
  - graph: [`qwen3-off-vs-compute-ccl-layer0.svg`](./qwen3-off-vs-compute-ccl-layer0.svg)

Regenerate:

```bash
python docs/triton-backend/qwen3-layer0-fused-modes/generate_qwen3_layer0_fused_modes.py
```
