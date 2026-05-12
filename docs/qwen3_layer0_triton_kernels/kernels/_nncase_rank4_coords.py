# Generated Triton helper extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2426-2441
# Note: helper kernels are shared by the per-op kernels in this folder.

import triton
import triton.language as tl

@triton.jit
def _nncase_rank4_coords(offsets, shape_table, pe):
    base = pe * 4
    n0 = tl.maximum(tl.load(shape_table + base + 0), 1)
    n1 = tl.maximum(tl.load(shape_table + base + 1), 1)
    n2 = tl.maximum(tl.load(shape_table + base + 2), 1)
    n3 = tl.maximum(tl.load(shape_table + base + 3), 1)
    i3 = offsets % n3
    tmp = offsets // n3
    i2 = tmp % n2
    tmp = tmp // n2
    i1 = tmp % n1
    i0 = tmp // n1
    return i0, i1, i2, i3
