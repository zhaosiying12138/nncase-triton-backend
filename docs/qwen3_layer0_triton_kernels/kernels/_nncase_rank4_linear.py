# Generated Triton helper extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2441-2450
# Note: helper kernels are shared by the per-op kernels in this folder.

import triton
import triton.language as tl

@triton.jit
def _nncase_rank4_linear(i0, i1, i2, i3, stride_table, pe):
    base = pe * 4
    s0 = tl.load(stride_table + base + 0)
    s1 = tl.load(stride_table + base + 1)
    s2 = tl.load(stride_table + base + 2)
    s3 = tl.load(stride_table + base + 3)
    return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3
