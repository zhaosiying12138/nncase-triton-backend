# Qwen3 decoder layer 0, ordinal 13: device_func_26240
# Source module: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py
# PE count captured in this export: 20
#
# Qwen3 meaning: k_norm + reshape/transpose/cast into [kv_heads=8,S,head_dim=128]
# nncase kind: function. This ordinal enters a generated nncase device function. The function may lower several inner nncase ops to several Triton kernels.
# Arguments: buffer_31, const_32, const_33, dim_var, buffer_34
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
# Kernel: _nncase_pe_layer_norm_transpose_rank4_kernel
# Implements here: fused RMSNorm plus transpose for Q/K head layout preparation.
# NUMA model: The local tensor shard is normalized and transposed inside the same PE-local GMEM region.
#==============================================================================================
# Generated Triton kernel extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2858-2930
# Used by decoder layer 0 ordinals: 5, 13
# Study note: Fused q_norm/k_norm + transpose/cast path for q/k heads.
# Study note: This avoids materializing a separate normalized tensor before head layout conversion.
# NUMA note: PE is encoded in a Triton grid dimension and selects pointer-table entries.

import triton
import triton.language as tl

# This file is for reading. In the generated module these helpers live in the same file.
# from common_helpers import _nncase_load_by_type, _nncase_store_by_type, _nncase_rank4_coords, _nncase_rank4_linear, _nncase_rank4_broadcast_linear

@triton.jit
def _nncase_pe_layer_norm_transpose_rank4_kernel(x_ptrs, scale_ptrs, bias_ptrs, out_ptrs,
                                                 row_table, x_shape_table, x_stride_table, out_stride_table,
                                                 p0:tl.constexpr, p1:tl.constexpr, p2:tl.constexpr, p3:tl.constexpr,
                                                 cols:tl.constexpr, eps:tl.constexpr, use_mean:tl.constexpr,
                                                 x_dtype:tl.constexpr, scale_dtype:tl.constexpr,
                                                 bias_dtype:tl.constexpr, out_dtype:tl.constexpr,
                                                 BLOCK:tl.constexpr):
    row = tl.program_id(0)
    pe = tl.program_id(1)
    rows = tl.load(row_table + pe)
    row_active = row < rows
    base = pe * 4
    n2 = tl.maximum(tl.load(x_shape_table + base + 2), 1)
    n1 = tl.maximum(tl.load(x_shape_table + base + 1), 1)
    i2 = row % n2
    tmp = row // n2
    i1 = tmp % n1
    i0 = tmp // n1
    cols_offsets = tl.arange(0, BLOCK)
    mask = (cols_offsets < cols) & row_active
    x_base = tl.load(x_ptrs + pe)
    scale_base = tl.load(scale_ptrs + pe)
    bias_base = tl.load(bias_ptrs + pe)
    out_base = tl.load(out_ptrs + pe)
    x_offsets = _nncase_rank4_linear(i0, i1, i2, cols_offsets, x_stride_table, pe)
    values = _nncase_load_by_type(x_base, x_offsets, mask, x_dtype).to(tl.float32)
    if use_mean:
        mean = tl.sum(values, axis=0) / cols
        centered = tl.where(mask, values - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / cols
        norm = centered * tl.rsqrt(var + eps)
    else:
        var = tl.sum(tl.where(mask, values * values, 0.0), axis=0) / cols
        norm = values * tl.rsqrt(var + eps)
    s = _nncase_load_by_type(scale_base, cols_offsets, mask, scale_dtype).to(tl.float32)
    b = _nncase_load_by_type(bias_base, cols_offsets, mask, bias_dtype).to(tl.float32)
    src0 = i0
    src1 = i1
    src2 = i2
    src3 = cols_offsets
    d0 = src0
    if p0 == 1:
        d0 = src1
    elif p0 == 2:
        d0 = src2
    elif p0 == 3:
        d0 = src3
    d1 = src0
    if p1 == 1:
        d1 = src1
    elif p1 == 2:
        d1 = src2
    elif p1 == 3:
        d1 = src3
    d2 = src0
    if p2 == 1:
        d2 = src1
    elif p2 == 2:
        d2 = src2
    elif p2 == 3:
        d2 = src3
    d3 = src0
    if p3 == 1:
        d3 = src1
    elif p3 == 2:
        d3 = src2
    elif p3 == 3:
        d3 = src3
    out_offsets = _nncase_rank4_linear(d0, d1, d2, d3, out_stride_table, pe)
    _nncase_store_by_type(out_base, out_offsets, norm * s + b, mask, out_dtype)
