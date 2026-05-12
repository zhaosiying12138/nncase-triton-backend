# Generated Triton kernel extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2714-2725
# Used by decoder layer 0 ordinals: 6
# Study note: Writes sequential position ids starting from the current KV-cache token offset.
# Study note: For decode this start value comes from the runtime paged KV state.
# NUMA note: PE is encoded in a Triton grid dimension and selects pointer-table entries.

import triton
import triton.language as tl

# This file is for reading. In the generated module these helpers live in the same file.
# from common_helpers import _nncase_load_by_type, _nncase_store_by_type, _nncase_rank4_coords, _nncase_rank4_linear, _nncase_rank4_broadcast_linear

@triton.jit
def _nncase_pe_position_ids_kernel(dst_ptrs, total_table, start:tl.constexpr, local_offsets, dst_dtype:tl.constexpr, BLOCK:tl.constexpr):
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    base = tl.load(dst_ptrs + pe)
    local_offset = tl.load(local_offsets + pe)
    _nncase_store_by_type(base, offsets, start + local_offset + offsets, mask, dst_dtype)
