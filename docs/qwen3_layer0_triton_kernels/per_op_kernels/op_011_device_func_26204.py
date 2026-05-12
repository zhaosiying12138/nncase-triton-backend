# Qwen3 decoder layer 0, ordinal 11: device_func_26204
# Source module: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py
# PE count captured in this export: 20
#
# Qwen3 meaning: attention k/v side 1024-wide projection from normalized hidden
# nncase kind: function. This ordinal enters a generated nncase device function. The function may lower several inner nncase ops to several Triton kernels.
# Arguments: buffer_12, const_27, dim_var, buffer_28
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
