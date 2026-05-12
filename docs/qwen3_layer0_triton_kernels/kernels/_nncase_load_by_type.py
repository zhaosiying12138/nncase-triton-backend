# Generated Triton helper extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2384-2405
# Note: helper kernels are shared by the per-op kernels in this folder.

import triton
import triton.language as tl

@triton.jit
def _nncase_load_by_type(base, offsets, mask, dtype_code:tl.constexpr):
    if dtype_code == 0:
        return tl.load(base.to(tl.pointer_type(tl.int8)) + offsets, mask=mask, other=0)
    elif dtype_code == 2:
        return tl.load(base.to(tl.pointer_type(tl.int8)) + offsets, mask=mask, other=0)
    elif dtype_code == 3:
        return tl.load(base.to(tl.pointer_type(tl.int16)) + offsets, mask=mask, other=0)
    elif dtype_code == 4:
        return tl.load(base.to(tl.pointer_type(tl.int32)) + offsets, mask=mask, other=0)
    elif dtype_code == 5:
        return tl.load(base.to(tl.pointer_type(tl.int64)) + offsets, mask=mask, other=0)
    elif dtype_code == 6:
        return tl.load(base.to(tl.pointer_type(tl.uint8)) + offsets, mask=mask, other=0)
    elif dtype_code == 10:
        return tl.load(base.to(tl.pointer_type(tl.float16)) + offsets, mask=mask, other=0.0)
    elif dtype_code == 13:
        return tl.load(base.to(tl.pointer_type(tl.bfloat16)) + offsets, mask=mask, other=0.0)
    else:
        return tl.load(base.to(tl.pointer_type(tl.float32)) + offsets, mask=mask, other=0.0)
