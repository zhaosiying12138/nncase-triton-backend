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
