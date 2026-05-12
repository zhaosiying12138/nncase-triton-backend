# Qwen3 decoder layer 0, ordinal 18: paged_attention
# Source module: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py
# PE count captured in this export: 20
#
# Qwen3 meaning: paged attention over the simulated per-PE KV cache
# nncase kind: collective. This ordinal is a collective-style op. It is still a Triton launch, but it intentionally reads PE ownership metadata.
# Arguments: buffer_26, kvCache, buffer_39, const_40, buffer_41
#
# How to read this study file:
# - The code below is copied from the generated Triton module for study only.
# - The real runnable copy stays in tests_output/.../triton_module.py.
# - Generic rank4 kernels pad tensors to 4 logical axes, then use descriptor-provided shapes/strides.
# - PE-local kernels launch one PE dimension in the Triton grid and select ptrs[pe].
# - Non-collective kernels should not read or write another PE NUMA region.
# - This collective kernel is the exception: it uses owner/slot tables to model cross-PE KV access.

import triton
import triton.language as tl

# Helper note: _nncase_load_by_type/_store_by_type and rank4 coordinate helpers are in ../kernels/.
# They are omitted here only to keep each op file focused on the generated kernels it directly invokes.

#==============================================================================================
# Kernel: _nncase_collective_paged_attention_rank3_kernel
# Implements here: paged attention over the sharded KV cache. This is the collective-style kernel in this layer.
# NUMA model: It consults owner/slot tables so the attention computation can read the PE shard that owns each cached token.
#==============================================================================================
# Generated Triton kernel extracted from nncase qwen3 PE=20 output.
# Source: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py:3049-3154
# Used by decoder layer 0 ordinals: 18
# Study note: Attention collective over per-PE KV cache shards.
# Study note: It uses owner_table/slot_table to read the PE that owns each cached token, which is the CCL-like NUMA access point.
# NUMA note: PE is encoded in a Triton grid dimension and selects pointer-table entries.

import triton
import triton.language as tl

# This file is for reading. In the generated module these helpers live in the same file.
# from common_helpers import _nncase_load_by_type, _nncase_store_by_type, _nncase_rank4_coords, _nncase_rank4_linear, _nncase_rank4_broadcast_linear

@triton.jit
def _nncase_collective_paged_attention_rank3_kernel(q_ptrs, key_ptrs, value_ptrs, out_ptrs, scale_ptrs,
                                                     q_shape_table, q_offset_table,
                                                     q_stride_table, key_stride_table, value_stride_table, out_stride_table,
                                                     owner_table, slot_table,
                                                     start:tl.constexpr, end:tl.constexpr, repeat:tl.constexpr,
                                                     q_dtype:tl.constexpr, key_dtype:tl.constexpr,
                                                     value_dtype:tl.constexpr, out_dtype:tl.constexpr, scale_dtype:tl.constexpr,
                                                     HEAD_DIM:tl.constexpr, BLOCK_T:tl.constexpr, BLOCK_D:tl.constexpr):
    pid = tl.program_id(0)
    q_head = pid // HEAD_DIM
    out_dim = pid - q_head * HEAD_DIM
    q_seq = tl.program_id(1)
    pe = tl.program_id(2)
    q_base_index = pe * 4
    local_heads = tl.load(q_shape_table + q_base_index + 1)
    local_seq = tl.load(q_shape_table + q_base_index + 3)
    active = (q_head < local_heads) & (q_seq < local_seq)
    q_head_global = q_head + tl.load(q_offset_table + q_base_index + 1)
    q_seq_global = q_seq + tl.load(q_offset_table + q_base_index + 3)
    query_pos = start + q_seq_global
    kv_head = q_head_global // repeat
    q_s1 = tl.load(q_stride_table + q_base_index + 1)
    q_s2 = tl.load(q_stride_table + q_base_index + 2)
    q_s3 = tl.load(q_stride_table + q_base_index + 3)
    out_s1 = tl.load(out_stride_table + q_base_index + 1)
    out_s2 = tl.load(out_stride_table + q_base_index + 2)
    out_s3 = tl.load(out_stride_table + q_base_index + 3)
    t = tl.arange(0, BLOCK_T)
    valid_t = t < end
    owner = tl.load(owner_table + t, mask=valid_t, other=-1)
    local_t = tl.load(slot_table + t, mask=valid_t, other=-1)
    local_t = tl.where(local_t < 0, t, local_t)
    src_pe = tl.where(owner < 0, pe, owner)
    key_base = tl.load(key_ptrs + src_pe, mask=valid_t, other=0)
    k_stride_index = src_pe * 4
    k_s1 = tl.load(key_stride_table + k_stride_index + 1, mask=valid_t, other=0)
    k_s2 = tl.load(key_stride_table + k_stride_index + 2, mask=valid_t, other=0)
    k_s3 = tl.load(key_stride_table + k_stride_index + 3, mask=valid_t, other=0)
    q_base = tl.load(q_ptrs + pe)
    scores = tl.zeros((BLOCK_T,), tl.float32)
    for d0 in range(0, HEAD_DIM, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        d_mask = d < HEAD_DIM
        q_offsets = q_head * q_s1 + d * q_s2 + q_seq * q_s3
        q_vals = _nncase_load_by_type(q_base, q_offsets, active & d_mask, q_dtype).to(tl.float32)
        k_offsets = kv_head * k_s1[None, :] + d[:, None] * k_s2[None, :] + local_t[None, :] * k_s3[None, :]
        k_vals = _nncase_load_by_type(key_base[None, :], k_offsets, d_mask[:, None] & valid_t[None, :], key_dtype).to(tl.float32)
        scores += tl.sum(k_vals * q_vals[:, None], axis=0)
    scale_base = tl.load(scale_ptrs + pe)
    scale = _nncase_load_by_type(scale_base, 0, True, scale_dtype).to(tl.float32)
    causal_mask = valid_t & (t <= query_pos) & active
    scores = tl.where(causal_mask, scores * scale, -3.4028234663852886e38)
    max_score = tl.max(scores, axis=0)
    probs = tl.exp(scores - max_score)
    probs = tl.where(causal_mask, probs, 0.0)
    denom = tl.sum(probs, axis=0)
    denom = tl.where(denom == 0.0, 1.0, denom)
    probs = probs / denom
    value_base = tl.load(value_ptrs + src_pe, mask=valid_t, other=0)
    v_stride_index = src_pe * 4
    v_s1 = tl.load(value_stride_table + v_stride_index + 1, mask=valid_t, other=0)
    v_s2 = tl.load(value_stride_table + v_stride_index + 2, mask=valid_t, other=0)
    v_s3 = tl.load(value_stride_table + v_stride_index + 3, mask=valid_t, other=0)
    v_offsets = kv_head * v_s1 + out_dim * v_s2 + local_t * v_s3
    v_vals = _nncase_load_by_type(value_base, v_offsets, valid_t & active, value_dtype).to(tl.float32)
    acc = tl.sum(probs * v_vals, axis=0)
    out_base = tl.load(out_ptrs + pe)
    out_offsets = q_head * out_s1 + out_dim * out_s2 + q_seq * out_s3
    _nncase_store_by_type(out_base, out_offsets, acc, active & (out_dim < HEAD_DIM), out_dtype)
