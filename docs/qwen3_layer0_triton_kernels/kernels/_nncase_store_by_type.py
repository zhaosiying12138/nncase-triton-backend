# Generated Triton helper extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2405-2426
# Note: helper kernels are shared by the per-op kernels in this folder.

import triton
import triton.language as tl

@triton.jit
def _nncase_store_by_type(base, offsets, values, mask, dtype_code:tl.constexpr):
    if dtype_code == 0:
        tl.store(base.to(tl.pointer_type(tl.int8)) + offsets, values, mask=mask)
    elif dtype_code == 2:
        tl.store(base.to(tl.pointer_type(tl.int8)) + offsets, values, mask=mask)
    elif dtype_code == 3:
        tl.store(base.to(tl.pointer_type(tl.int16)) + offsets, values, mask=mask)
    elif dtype_code == 4:
        tl.store(base.to(tl.pointer_type(tl.int32)) + offsets, values, mask=mask)
    elif dtype_code == 5:
        tl.store(base.to(tl.pointer_type(tl.int64)) + offsets, values, mask=mask)
    elif dtype_code == 6:
        tl.store(base.to(tl.pointer_type(tl.uint8)) + offsets, values, mask=mask)
    elif dtype_code == 10:
        tl.store(base.to(tl.pointer_type(tl.float16)) + offsets, values, mask=mask)
    elif dtype_code == 13:
        tl.store(base.to(tl.pointer_type(tl.bfloat16)) + offsets, values, mask=mask)
    else:
        tl.store(base.to(tl.pointer_type(tl.float32)) + offsets, values, mask=mask)
