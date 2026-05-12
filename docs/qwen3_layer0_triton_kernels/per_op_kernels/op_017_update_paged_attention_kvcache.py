# Qwen3 decoder layer 0, ordinal 17: update_paged_attention_kvcache
# Source module: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py
# PE count captured in this export: 20
#
# Qwen3 meaning: write value shard into simulated NUMA KV cache according to slot_mapping owner/local_slot
# nncase kind: compute. This ordinal is a direct compute op lowered to one Triton kernel family.
# Arguments: buffer_30, kvCache
#
# How to read this study file:
# - The code below is copied from the generated Triton module for study only.
# - The real runnable copy stays in tests_output/.../triton_module.py.
# - Generic rank4 kernels pad tensors to 4 logical axes, then use descriptor-provided shapes/strides.
# - PE-local kernels launch one PE dimension in the Triton grid and select ptrs[pe].
# - Non-collective kernels should not read or write another PE NUMA region.

import triton
import triton.language as tl

# Helper note: _nncase_load_by_type/_store_by_type and rank4 coordinate helpers are in ../kernels/.
# They are omitted here only to keep each op file focused on the generated kernels it directly invokes.

#==============================================================================================
# Kernel: _nncase_pe_update_kv_rank4_kernel
# Implements here: writes generated key/value vectors into the simulated per-PE paged KV cache.
# NUMA model: slot_mapping chooses owner PE and local slot; only owner == pe performs the store.
#==============================================================================================
# Generated Triton kernel extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:2725-2751
# Used by decoder layer 0 ordinals: 16, 17
# Study note: Writes K or V slots into the simulated NUMA KV cache.
# Study note: When slot_mapping is present, only owner == pe stores, and local_slot chooses the PE-local cache index.
# NUMA note: PE is encoded in a Triton grid dimension and selects pointer-table entries.

import triton
import triton.language as tl

# This file is for reading. In the generated module these helpers live in the same file.
# from common_helpers import _nncase_load_by_type, _nncase_store_by_type, _nncase_rank4_coords, _nncase_rank4_linear, _nncase_rank4_broadcast_linear

@triton.jit
def _nncase_pe_update_kv_rank4_kernel(slots_ptrs, cache_ptrs,
                                       total_table, slots_shape_table, slots_stride_table, cache_stride_table,
                                       seq_offsets, slot_mapping,
                                       start:tl.constexpr, slot_stride0:tl.constexpr, slot_stride1:tl.constexpr,
                                       use_slot_mapping:tl.constexpr,
                                       slots_dtype:tl.constexpr, cache_dtype:tl.constexpr,
                                       BLOCK:tl.constexpr):
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    i0, i1, i2, i3 = _nncase_rank4_coords(offsets, slots_shape_table, pe)
    src_offsets = _nncase_rank4_linear(i0, i1, i2, i3, slots_stride_table, pe)
    dst_i3 = i3 + start + tl.load(seq_offsets + pe)
    store_mask = mask
    if use_slot_mapping:
        owner = tl.load(slot_mapping + i3 * slot_stride0, mask=mask, other=-1)
        local_slot = tl.load(slot_mapping + i3 * slot_stride0 + slot_stride1, mask=mask, other=-1)
        dst_i3 = local_slot
        store_mask = mask & (owner == pe) & (local_slot >= 0)
    dst_offsets = _nncase_rank4_linear(i0, i1, i2, dst_i3, cache_stride_table, pe)
    values = _nncase_load_by_type(tl.load(slots_ptrs + pe), src_offsets, mask, slots_dtype)
    _nncase_store_by_type(tl.load(cache_ptrs + pe), dst_offsets, values, store_mask, cache_dtype)
