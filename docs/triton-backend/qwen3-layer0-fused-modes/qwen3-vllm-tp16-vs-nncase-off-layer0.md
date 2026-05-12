# Qwen3 vLLM TP=16 vs nncase `--fused-kernel=off` Layer0 Graph

This graph compares two different shard interpretations for Qwen3-0.6B layer0:

- left: vLLM `tensor_parallel_size=16`, shown as one TP rank's local graph;
- middle: one per-row mapping/difference note;
- right: nncase CUDA/Triton `PE=16 --fused-kernel=off`, using one node per
  layer0 ordinal, matching the `qwen3-off-layer0.svg` convention.

Source artifacts:

- `docs/triton-backend/qwen3-vllm-tp16-vs-nncase-off.md`
- `docs/triton-backend/qwen3-layer0-fused-modes/qwen3-off-layer0.md`
- `tests_output/qwen3_reshard_off_probe2/cuda_admission/pe_16/CodeGen/cuda/cuda_meta.json`
- `tests/llm/Qwen/Qwen3-0.6B/config.json`
- vLLM Qwen3 source: <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py>
- vLLM Qwen2 MLP source: <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2.py>
- vLLM parallel linear source: <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py>

Metadata checks:

- Qwen3-0.6B: `H=1024`, `I=3072`, `QH=16`, `KVH=8`, `D=128`
- vLLM TP=16: each rank owns `1` Q head; K/V heads are `2`-way replicated
  because `KVH=8 < TP=16`
- nncase off graph: `pe_count=16`, `fused_kernel=off`, layer0 ordinals
  `0..33`, with `34` launch records and `11` `requires_collective=true`
  records

## Graph

Render command:

```bash
dot -Tsvg \
  docs/triton-backend/qwen3-layer0-fused-modes/qwen3-vllm-tp16-vs-nncase-off-layer0.dot \
  -o docs/triton-backend/qwen3-layer0-fused-modes/qwen3-vllm-tp16-vs-nncase-off-layer0.svg
```

<img src="qwen3-vllm-tp16-vs-nncase-off-layer0.svg" alt="Qwen3 vLLM TP16 vs nncase off layer0 graph" style="width: 100%; height: auto;">

## Reading Notes

- The left column is source-derived from vLLM, not a local vLLM benchmark.
- The right column keeps one nncase node per ordinal, so CCL insertion points
  stay visible instead of being grouped into phases.
- The middle column is the intended reading path: each row says whether the two
  sides are the same logical op, a shard-axis difference, a fusion difference,
  or a CCL placement difference.
- Blue diamonds are vLLM TP collectives. Pink diamonds are nncase
  `requires_collective=true` ordinals.
- vLLM physically produces Q/K/V with one packed `QKVParallelLinear`; the graph
  splits Q, K, and V into logical rows because the shard semantics differ.

Key differences visible in the rows:

| Area | vLLM TP=16 | nncase `off`, PE=16 |
| --- | --- | --- |
| Embedding | vocab shard, then TP all-reduce to replicated hidden | hidden/SBP shard, then ord04 all-to-all `(B,S0) -> (S0,B)` |
| Input norm / K | RMSNorm and packed `k_proj` are separate logical rows | ord05 fuses materialization, input layernorm, and `k_proj` |
| Q path | no CCL before `q_proj`; each rank owns 1 Q head | ord06 all-gathers hidden, then ord07 emits `(B,S0)` Q |
| RoPE tables | cos/sin are local per TP rank | ord11/ord13 all-gather cos/sin |
| V path | no CCL before `v_proj`; one replicated KV head/rank | ord16 all-to-all before V matmul; ord17 is partial |
| Attention | local attention over local Q and replicated K/V | ord24 is a collective `paged_attention` |
| O projection | row-parallel partial, then TP all-reduce | ord26 partial, then ord27 reduce-scatter |
| Post attention | residual/post norm stay replicated | ord29 and ord30 all-gather residual/post_norm |
| MLP | gate/up local shards, down partial, then TP all-reduce before residual | ord31 fuses gate/up/act/mul; ord32 produces partial; ord33 consumes it at boundary |
