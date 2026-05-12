# Generated Triton kernel extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2521-2545
# Used by decoder layer 0 ordinals: 7, 21, 25, 26
# Study note: Generic binary elementwise kernel. op_code selects add, mul, or compare.equal.
# Study note: Broadcasting is driven by per-PE shape/stride tables, not by torch.
# NUMA note: PE is encoded in a Triton grid dimension and selects pointer-table entries.

import triton
import triton.language as tl

# This file is for reading. In the generated module these helpers live in the same file.
# from common_helpers import _nncase_load_by_type, _nncase_store_by_type, _nncase_rank4_coords, _nncase_rank4_linear, _nncase_rank4_broadcast_linear

@triton.jit
def _nncase_pe_binary_rank4_kernel(lhs_ptrs, rhs_ptrs, dst_ptrs, total_table, shape_table,
                                    lhs_shape_table, rhs_shape_table,
                                    lhs_stride_table, rhs_stride_table, dst_stride_table,
                                    op_code:tl.constexpr, lhs_dtype:tl.constexpr, rhs_dtype:tl.constexpr,
                                    dst_dtype:tl.constexpr, BLOCK:tl.constexpr):
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    i0, i1, i2, i3 = _nncase_rank4_coords(offsets, shape_table, pe)
    lhs_offsets = _nncase_rank4_broadcast_linear(i0, i1, i2, i3, lhs_shape_table, lhs_stride_table, pe)
    rhs_offsets = _nncase_rank4_broadcast_linear(i0, i1, i2, i3, rhs_shape_table, rhs_stride_table, pe)
    dst_offsets = _nncase_rank4_linear(i0, i1, i2, i3, dst_stride_table, pe)
    left = _nncase_load_by_type(tl.load(lhs_ptrs + pe), lhs_offsets, mask, lhs_dtype)
    right = _nncase_load_by_type(tl.load(rhs_ptrs + pe), rhs_offsets, mask, rhs_dtype)
    y = left + right
    if op_code == 1:
        y = left * right
    elif op_code == 2:
        y = left == right
    _nncase_store_by_type(tl.load(dst_ptrs + pe), dst_offsets, y, mask, dst_dtype)
