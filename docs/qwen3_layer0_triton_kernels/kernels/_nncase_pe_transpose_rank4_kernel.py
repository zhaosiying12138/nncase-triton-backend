# Generated Triton kernel extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2635-2687
# Used by decoder layer 0 ordinals: 10, 12, 15, 19
# Study note: Generic PE-local transpose kernel, also able to cast while storing by dst dtype.
# Study note: Used to switch between [S,H,D] and [H,D,S] layouts around attention/KV cache.
# NUMA note: PE is encoded in a Triton grid dimension and selects pointer-table entries.

import triton
import triton.language as tl

# This file is for reading. In the generated module these helpers live in the same file.
# from common_helpers import _nncase_load_by_type, _nncase_store_by_type, _nncase_rank4_coords, _nncase_rank4_linear, _nncase_rank4_broadcast_linear

@triton.jit
def _nncase_pe_transpose_rank4_kernel(src_ptrs, dst_ptrs, total_table, dst_shape_table,
                                       src_stride_table, dst_stride_table,
                                       p0:tl.constexpr, p1:tl.constexpr, p2:tl.constexpr, p3:tl.constexpr,
                                       src_dtype:tl.constexpr, dst_dtype:tl.constexpr, BLOCK:tl.constexpr):
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    i0, i1, i2, i3 = _nncase_rank4_coords(offsets, dst_shape_table, pe)
    s0 = tl.zeros((BLOCK,), tl.int64)
    s1 = tl.zeros((BLOCK,), tl.int64)
    s2 = tl.zeros((BLOCK,), tl.int64)
    s3 = tl.zeros((BLOCK,), tl.int64)
    if p0 == 0:
        s0 = i0
    elif p0 == 1:
        s1 = i0
    elif p0 == 2:
        s2 = i0
    else:
        s3 = i0
    if p1 == 0:
        s0 = i1
    elif p1 == 1:
        s1 = i1
    elif p1 == 2:
        s2 = i1
    else:
        s3 = i1
    if p2 == 0:
        s0 = i2
    elif p2 == 1:
        s1 = i2
    elif p2 == 2:
        s2 = i2
    else:
        s3 = i2
    if p3 == 0:
        s0 = i3
    elif p3 == 1:
        s1 = i3
    elif p3 == 2:
        s2 = i3
    else:
        s3 = i3
    src_offsets = _nncase_rank4_linear(s0, s1, s2, s3, src_stride_table, pe)
    dst_offsets = _nncase_rank4_linear(i0, i1, i2, i3, dst_stride_table, pe)
    values = _nncase_load_by_type(tl.load(src_ptrs + pe), src_offsets, mask, src_dtype)
    _nncase_store_by_type(tl.load(dst_ptrs + pe), dst_offsets, values, mask, dst_dtype)
