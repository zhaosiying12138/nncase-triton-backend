# Generated Triton kernel extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2687-2714
# Used by decoder layer 0 ordinals: 9, 14
# Study note: Applies rotary embedding using generated cos/sin buffers.
# Study note: The kernel rotates the first and second half of each head_dim vector.
# NUMA note: PE is encoded in a Triton grid dimension and selects pointer-table entries.

import triton
import triton.language as tl

# This file is for reading. In the generated module these helpers live in the same file.
# from common_helpers import _nncase_load_by_type, _nncase_store_by_type, _nncase_rank4_coords, _nncase_rank4_linear, _nncase_rank4_broadcast_linear

@triton.jit
def _nncase_pe_rope_rank4_kernel(x_ptrs, cos_ptrs, sin_ptrs, out_ptrs,
                                  total_table, out_shape_table,
                                  cos_shape_table, sin_shape_table,
                                  x_stride_table, cos_stride_table, sin_stride_table, out_stride_table,
                                  half_dim:tl.constexpr,
                                  x_dtype:tl.constexpr, cos_dtype:tl.constexpr, sin_dtype:tl.constexpr,
                                  out_dtype:tl.constexpr, BLOCK:tl.constexpr):
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    i0, i1, i2, i3 = _nncase_rank4_coords(offsets, out_shape_table, pe)
    partner_dim = tl.where(i3 < half_dim, i3 + half_dim, i3 - half_dim)
    x_offsets = _nncase_rank4_linear(i0, i1, i2, i3, x_stride_table, pe)
    partner_offsets = _nncase_rank4_linear(i0, i1, i2, partner_dim, x_stride_table, pe)
    cos_offsets = _nncase_rank4_broadcast_linear(i0, i1, i2, i3, cos_shape_table, cos_stride_table, pe)
    sin_offsets = _nncase_rank4_broadcast_linear(i0, i1, i2, i3, sin_shape_table, sin_stride_table, pe)
    out_offsets = _nncase_rank4_linear(i0, i1, i2, i3, out_stride_table, pe)
    x = _nncase_load_by_type(tl.load(x_ptrs + pe), x_offsets, mask, x_dtype).to(tl.float32)
    partner = _nncase_load_by_type(tl.load(x_ptrs + pe), partner_offsets, mask, x_dtype).to(tl.float32)
    cos = _nncase_load_by_type(tl.load(cos_ptrs + pe), cos_offsets, mask, cos_dtype).to(tl.float32)
    sin = _nncase_load_by_type(tl.load(sin_ptrs + pe), sin_offsets, mask, sin_dtype).to(tl.float32)
    rotated = tl.where(i3 < half_dim, x * cos - partner * sin, x * cos + partner * sin)
    _nncase_store_by_type(tl.load(out_ptrs + pe), out_offsets, rotated, mask, out_dtype)
