# Generated Triton kernel extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2499-2521
# Used by decoder layer 0 ordinals: 7, 8, 24
# Study note: Generic unary elementwise kernel. op_code selects cos, sin, or swish/SiLU.
# Study note: The rank4 metadata lets one implementation serve 1D/2D/3D/4D generated tensors.
# NUMA note: PE is encoded in a Triton grid dimension and selects pointer-table entries.

import triton
import triton.language as tl

# This file is for reading. In the generated module these helpers live in the same file.
# from common_helpers import _nncase_load_by_type, _nncase_store_by_type, _nncase_rank4_coords, _nncase_rank4_linear, _nncase_rank4_broadcast_linear

@triton.jit
def _nncase_pe_unary_rank4_kernel(src_ptrs, dst_ptrs, total_table, shape_table, src_stride_table, dst_stride_table,
                                   op_code:tl.constexpr, src_dtype:tl.constexpr, dst_dtype:tl.constexpr,
                                   beta:tl.constexpr, BLOCK:tl.constexpr):
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    i0, i1, i2, i3 = _nncase_rank4_coords(offsets, shape_table, pe)
    src_offsets = _nncase_rank4_linear(i0, i1, i2, i3, src_stride_table, pe)
    dst_offsets = _nncase_rank4_linear(i0, i1, i2, i3, dst_stride_table, pe)
    x = _nncase_load_by_type(tl.load(src_ptrs + pe), src_offsets, mask, src_dtype).to(tl.float32)
    y = x
    if op_code == 1:
        y = tl.cos(x)
    elif op_code == 2:
        y = tl.sin(x)
    elif op_code == 3:
        y = x * tl.sigmoid(x * beta)
    _nncase_store_by_type(tl.load(dst_ptrs + pe), dst_offsets, y, mask, dst_dtype)
