# Qwen3 decoder layer 0, ordinal 21: device_func_26257
# Source module: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py
# PE count captured in this export: 20
#
# Qwen3 meaning: attention residual add
# nncase kind: function. This ordinal enters a generated nncase device function. The function may lower several inner nncase ops to several Triton kernels.
# Arguments: buffer_11, buffer_45, dim_var, buffer_46
#
# How to read this study file:
# - The code below is copied from the generated Triton module for study only.
# - The real runnable copy stays in tests_output/.../triton_module.py.
# - Generic rank4 kernels pad tensors to 4 logical axes, then use descriptor-provided shapes/strides.
# - PE-local kernels launch one PE dimension in the Triton grid and select ptrs[pe].
# - Non-collective kernels should not read or write another PE NUMA region.

import triton
import triton.language as tl

# Helper note: _nncase_load_by_type/_store_by_type and rank4 coordinate helpers are in ../kernels/.
# They are omitted here only to keep each op file focused on the generated kernels it directly invokes.

#==============================================================================================
# Kernel: _nncase_pe_binary_rank4_kernel
# Implements here: rank4-padded elementwise binary op such as add/mul/equal.
# NUMA model: Broadcasting is local to the PE shard; Partial inputs are reduced before this kernel when needed.
#==============================================================================================
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
