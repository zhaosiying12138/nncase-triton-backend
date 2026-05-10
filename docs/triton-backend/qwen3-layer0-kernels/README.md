# Qwen3 Layer 0 Triton Kernel Study Pack

This directory extracts the Triton kernels used by Qwen3-0.6B layer 0 in the
shard-16 CUDA/Triton PoC.

Source evidence:

- Generated module:
  `tests_output/test_qwen3_cuda_poc/cuda_admission/pe_16/CodeGen/cuda/triton_module.py`
- Verbose run log:
  `tests_output/qwen3_triton_3tok_verbose.log`
- Layer graph:
  `docs/triton-backend/qwen3-layer0-shard16.md`

Files:

- [`layer0_kernel_launches.md`](layer0_kernel_launches.md): ordinal to kernel
  launch map for layer 0.
- [`layer0_triton_kernels_annotated.py`](layer0_triton_kernels_annotated.py):
  annotated learning copy of the Triton kernels used by layer 0.

The annotated Python file is not imported by nncase. It is a study artifact.
The real generated module remains the source of execution.

## Reading Order

1. Start with `_nncase_rank4_coords`, `_nncase_rank4_linear`, and
   `_nncase_rank4_broadcast_linear`. These explain how flat tile offsets become
   tensor coordinates.
2. Read `_nncase_pe_layer_norm_kernel` and `_nncase_pe_matmul_kernel`. These are
   the core PE-local compute kernels.
3. Read `_nncase_ccl_rank4_kernel`. This is the main explicit NUMA
   communication primitive.
4. Read `_nncase_collective_paged_attention_rank3_kernel`. This is the only
   attention-specific collective kernel in layer 0.

The rule to keep in mind:

```text
Non-CCL kernels:
  pe = tl.program_id(last_grid_axis)
  base = tl.load(pointer_table + pe)
  only this PE-local pointer is accessed.

CCL / collective kernels:
  may read pointer_table + other_pe intentionally,
  because nncase inserted an explicit communication op.
```
