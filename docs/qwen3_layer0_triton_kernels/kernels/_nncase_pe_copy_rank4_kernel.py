# Generated Triton kernel extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2463-2477
# Used by decoder layer 0 ordinals: 25
# Study note: PE-local generic copy/cast kernel. grid y dimension is PE id.
# Study note: Reads src_ptrs[pe] and writes dst_ptrs[pe], so it stays inside one simulated NUMA domain.
# NUMA note: PE is encoded in a Triton grid dimension and selects pointer-table entries.

import triton
import triton.language as tl

# This file is for reading. In the generated module these helpers live in the same file.
# from common_helpers import _nncase_load_by_type, _nncase_store_by_type, _nncase_rank4_coords, _nncase_rank4_linear, _nncase_rank4_broadcast_linear

@triton.jit
def _nncase_pe_copy_rank4_kernel(src_ptrs, dst_ptrs, total_table, shape_table, src_stride_table, dst_stride_table,
                                  src_dtype:tl.constexpr, dst_dtype:tl.constexpr, BLOCK:tl.constexpr):
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    i0, i1, i2, i3 = _nncase_rank4_coords(offsets, shape_table, pe)
    src_offsets = _nncase_rank4_linear(i0, i1, i2, i3, src_stride_table, pe)
    dst_offsets = _nncase_rank4_linear(i0, i1, i2, i3, dst_stride_table, pe)
    values = _nncase_load_by_type(tl.load(src_ptrs + pe), src_offsets, mask, src_dtype)
    _nncase_store_by_type(tl.load(dst_ptrs + pe), dst_offsets, values, mask, dst_dtype)
