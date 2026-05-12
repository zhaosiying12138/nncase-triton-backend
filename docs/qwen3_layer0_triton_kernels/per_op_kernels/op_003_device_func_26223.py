# Qwen3 decoder layer 0, ordinal 3: device_func_26223
# Source module: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py
# PE count captured in this export: 20
#
# Qwen3 meaning: prefill/decode mask where + attention input RMSNorm + one 1024-wide attention projection; also preserves the residual tensor for the later add
# nncase kind: function. This ordinal enters a generated nncase device function. The function may lower several inner nncase ops to several Triton kernels.
# Arguments: const_0, const_1, const_2, buffer_6, const_7, buffer_9, dim_var, buffer_10, buffer_11, buffer_12
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
# Kernel: _nncase_pe_where_rank4_kernel
# Implements here: broadcast where/select. In layer 0 it builds the attention mask/value stream used before RMSNorm.
# NUMA model: The PE id is a Triton grid dimension; every PE reads cond/lhs/rhs through pointer tables and writes only dst_ptrs[pe].
#==============================================================================================
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

#==============================================================================================
# Kernel: _nncase_pe_layer_norm_kernel
# Implements here: RMSNorm/layer-norm over the last hidden dimension. It reduces one row inside a Triton block.
# NUMA model: Rows are PE-local rows; scale and bias are broadcast parameters, while x/out pointers are PE-local.
#==============================================================================================
# Generated Triton kernel extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2828-2858
# Used by decoder layer 0 ordinals: 3, 22
# Study note: RMSNorm/layer-norm over the last axis. Qwen uses use_mean=False for RMSNorm.
# Study note: One Triton program handles one row on one PE; BLOCK is next power of two of cols.
# NUMA note: PE is encoded in a Triton grid dimension and selects pointer-table entries.

import triton
import triton.language as tl

# This file is for reading. In the generated module these helpers live in the same file.
# from common_helpers import _nncase_load_by_type, _nncase_store_by_type, _nncase_rank4_coords, _nncase_rank4_linear, _nncase_rank4_broadcast_linear

@triton.jit
def _nncase_pe_layer_norm_kernel(x_ptrs, scale_ptrs, bias_ptrs, out_ptrs,
                                  row_table, cols:tl.constexpr,
                                  eps:tl.constexpr, use_mean:tl.constexpr,
                                  x_dtype:tl.constexpr, scale_dtype:tl.constexpr,
                                  bias_dtype:tl.constexpr, out_dtype:tl.constexpr,
                                  BLOCK:tl.constexpr):
    row = tl.program_id(0)
    pe = tl.program_id(1)
    rows = tl.load(row_table + pe)
    row_active = row < rows
    cols_offsets = tl.arange(0, BLOCK)
    mask = (cols_offsets < cols) & row_active
    x_base = tl.load(x_ptrs + pe)
    scale_base = tl.load(scale_ptrs + pe)
    bias_base = tl.load(bias_ptrs + pe)
    out_base = tl.load(out_ptrs + pe)
    values = _nncase_load_by_type(x_base, row * cols + cols_offsets, mask, x_dtype).to(tl.float32)
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
    _nncase_store_by_type(out_base, row * cols + cols_offsets, norm * s + b, mask, out_dtype)

#==============================================================================================
# Kernel: _nncase_pe_matmul_kernel
# Implements here: tiled matrix multiplication used by attention projections and MLP projections.
# NUMA model: program_id(2) is PE. It selects lhs/rhs/out pointer-table entries for that PE and never indexes another PE output.
#==============================================================================================
# Generated Triton kernel extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2751-2786
# Used by decoder layer 0 ordinals: 3, 4, 11, 20, 23, 25
# Study note: PE-local tiled matmul. program_id(2) is the PE id; program_id(0/1) tile M/N.
# Study note: It uses tl.dot over BLOCK_K and stores only to out_ptrs[pe].
# NUMA note: PE is encoded in a Triton grid dimension and selects pointer-table entries.

import triton
import triton.language as tl

# This file is for reading. In the generated module these helpers live in the same file.
# from common_helpers import _nncase_load_by_type, _nncase_store_by_type, _nncase_rank4_coords, _nncase_rank4_linear, _nncase_rank4_broadcast_linear

@triton.jit
def _nncase_pe_matmul_kernel(lhs_ptrs, rhs_ptrs, out_ptrs,
                             m_table, n_table, K:tl.constexpr,
                             lhs_stride_table, rhs_stride_table, out_stride_table,
                             lhs_dtype:tl.constexpr, rhs_dtype:tl.constexpr, out_dtype:tl.constexpr,
                             BLOCK_M:tl.constexpr, BLOCK_N:tl.constexpr, BLOCK_K:tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pe = tl.program_id(2)
    m = tl.load(m_table + pe)
    n = tl.load(n_table + pe)
    stride_base = pe * 4
    lhs_s0 = tl.load(lhs_stride_table + stride_base + 2)
    lhs_s1 = tl.load(lhs_stride_table + stride_base + 3)
    rhs_s0 = tl.load(rhs_stride_table + stride_base + 2)
    rhs_s1 = tl.load(rhs_stride_table + stride_base + 3)
    out_s0 = tl.load(out_stride_table + stride_base + 2)
    out_s1 = tl.load(out_stride_table + stride_base + 3)
    lhs_base = tl.load(lhs_ptrs + pe)
    rhs_base = tl.load(rhs_ptrs + pe)
    out_base = tl.load(out_ptrs + pe)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a_offsets = offs_m[:, None] * lhs_s0 + k[None, :] * lhs_s1
        b_offsets = k[:, None] * rhs_s0 + offs_n[None, :] * rhs_s1
        a = _nncase_load_by_type(lhs_base, a_offsets, (offs_m[:, None] < m) & (k[None, :] < K), lhs_dtype)
        b = _nncase_load_by_type(rhs_base, b_offsets, (k[:, None] < K) & (offs_n[None, :] < n), rhs_dtype)
        acc += tl.dot(a, b)
    out_offsets = offs_m[:, None] * out_s0 + offs_n[None, :] * out_s1
    _nncase_store_by_type(out_base, out_offsets, acc, (offs_m[:, None] < m) & (offs_n[None, :] < n), out_dtype)
