# Generated Triton kernel extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2565-2586
# Used by decoder layer 0 ordinals: 3
# Study note: Implements elementwise where for the attention input mask path.
# Study note: Condition, lhs, rhs, and dst may each have different broadcast strides.
# NUMA note: PE is encoded in a Triton grid dimension and selects pointer-table entries.

import triton
import triton.language as tl

# This file is for reading. In the generated module these helpers live in the same file.
# from common_helpers import _nncase_load_by_type, _nncase_store_by_type, _nncase_rank4_coords, _nncase_rank4_linear, _nncase_rank4_broadcast_linear

@triton.jit
def _nncase_pe_where_rank4_kernel(cond_ptrs, lhs_ptrs, rhs_ptrs, dst_ptrs, total_table, shape_table,
                                   cond_shape_table, lhs_shape_table, rhs_shape_table,
                                   cond_stride_table, lhs_stride_table, rhs_stride_table, dst_stride_table,
                                   cond_dtype:tl.constexpr, lhs_dtype:tl.constexpr, rhs_dtype:tl.constexpr,
                                   dst_dtype:tl.constexpr, BLOCK:tl.constexpr):
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    i0, i1, i2, i3 = _nncase_rank4_coords(offsets, shape_table, pe)
    cond_offsets = _nncase_rank4_broadcast_linear(i0, i1, i2, i3, cond_shape_table, cond_stride_table, pe)
    lhs_offsets = _nncase_rank4_broadcast_linear(i0, i1, i2, i3, lhs_shape_table, lhs_stride_table, pe)
    rhs_offsets = _nncase_rank4_broadcast_linear(i0, i1, i2, i3, rhs_shape_table, rhs_stride_table, pe)
    dst_offsets = _nncase_rank4_linear(i0, i1, i2, i3, dst_stride_table, pe)
    pred = _nncase_load_by_type(tl.load(cond_ptrs + pe), cond_offsets, mask, cond_dtype) != 0
    left = _nncase_load_by_type(tl.load(lhs_ptrs + pe), lhs_offsets, mask, lhs_dtype)
    right = _nncase_load_by_type(tl.load(rhs_ptrs + pe), rhs_offsets, mask, rhs_dtype)
    _nncase_store_by_type(tl.load(dst_ptrs + pe), dst_offsets, tl.where(pred, left, right), mask, dst_dtype)
