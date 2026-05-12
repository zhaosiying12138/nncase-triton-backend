# Qwen3 Layer 0 Triton Kernels

This folder is a study-oriented export from the current generated CUDA/Triton module. It does not replace the generated runtime file.

## What Was Exported

- Source module: `tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py`
- PE count in source metadata: `20`
- Function: `main_segment_1_prim`
- Decoder-layer boundary: ordinals `3..26`
- Ordinals `0..2` are input load / input condition / embedding gather and are intentionally excluded from this layer folder.

## How To Read This Folder

- `op_sequence_layer0.md` is the main index. Start there to see the Qwen3-level op order and which Triton kernels each generated op uses.
- `ops/op_*.md` has one file per generated launch, including inner nncase ops for generated device functions.
- `kernels/*.py` contains the extracted generated Triton `@triton.jit` bodies plus comments. These files are for reading; the real runnable copy remains in the generated module.
- `source_manifest.json` records the source path, ordinal range, and kernel/op mapping used by this export.

## PE/NUMA Interpretation

- For PE-local kernels, `tl.program_id(1)` or `tl.program_id(2)` is the simulated PE id. That PE selects `ptrs[pe]`, so the kernel reads and writes the GMEM region assigned to that PE.
- For matmul, the grid is `(M tiles, N tiles, PE)`; for elementwise/transpose/RoPE/update-KV it is usually `(linear tiles, PE)`.
- `update_paged_attention_kvcache` uses the paged KV `slot_mapping`: only the PE whose id equals `owner` writes the slot, and `local_slot` is the PE-local cache index.
- `paged_attention` is the collective-like attention kernel: it consults owner/slot tables to read the cache shard that owns each token.

## Kernel Files

- `kernels/_nncase_collective_paged_attention_rank3_kernel.py`: used by ordinals 18.
- `kernels/_nncase_pe_binary_rank4_kernel.py`: used by ordinals 7, 21, 25, 26.
- `kernels/_nncase_pe_copy_rank4_kernel.py`: used by ordinals 25.
- `kernels/_nncase_pe_layer_norm_kernel.py`: used by ordinals 3, 22.
- `kernels/_nncase_pe_layer_norm_transpose_rank4_kernel.py`: used by ordinals 5, 13.
- `kernels/_nncase_pe_matmul_kernel.py`: used by ordinals 3, 4, 11, 20, 23, 25.
- `kernels/_nncase_pe_position_ids_kernel.py`: used by ordinals 6.
- `kernels/_nncase_pe_rope_rank4_kernel.py`: used by ordinals 9, 14.
- `kernels/_nncase_pe_transpose_rank4_kernel.py`: used by ordinals 10, 12, 15, 19.
- `kernels/_nncase_pe_unary_rank4_kernel.py`: used by ordinals 7, 8, 24.
- `kernels/_nncase_pe_update_kv_rank4_kernel.py`: used by ordinals 16, 17.
- `kernels/_nncase_pe_where_rank4_kernel.py`: used by ordinals 3.
