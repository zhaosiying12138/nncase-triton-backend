# Generated Triton helper extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2450-2463
# Note: helper kernels are shared by the per-op kernels in this folder.

import triton
import triton.language as tl

@triton.jit
def _nncase_rank4_broadcast_linear(i0, i1, i2, i3, shape_table, stride_table, pe):
    base = pe * 4
    n0 = tl.load(shape_table + base + 0)
    n1 = tl.load(shape_table + base + 1)
    n2 = tl.load(shape_table + base + 2)
    n3 = tl.load(shape_table + base + 3)
    b0 = tl.where(n0 == 1, 0, i0)
    b1 = tl.where(n1 == 1, 0, i1)
    b2 = tl.where(n2 == 1, 0, i2)
    b3 = tl.where(n3 == 1, 0, i3)
    return _nncase_rank4_linear(b0, b1, b2, b3, stride_table, pe)
