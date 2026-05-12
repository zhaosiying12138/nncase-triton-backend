# Fused PE-local MLP kernel used by Qwen3 decoder layer 0 ordinal 25.
#
# Computes:
#   up = lhs @ up_rhs
#   out = (gate * up) @ down_rhs
#
# `gate` is already the SiLU-activated gate branch from the previous ordinal.
# program_id(2) is the simulated PE id. All pointer tables are indexed by that
# PE id, so the normal compute path stays inside one NUMA GMEM partition.

import triton
import triton.language as tl


@triton.jit
def _nncase_pe_matmul_mul_matmul_kernel(gate_ptrs, lhs_ptrs, up_rhs_ptrs, down_rhs_ptrs, out_ptrs,
                                        m_table, n_table, mid_table,
                                        K_LHS: tl.constexpr, K_MID: tl.constexpr,
                                        gate_stride_table, lhs_stride_table, up_rhs_stride_table, down_rhs_stride_table, out_stride_table,
                                        gate_dtype: tl.constexpr, lhs_dtype: tl.constexpr, up_rhs_dtype: tl.constexpr, down_rhs_dtype: tl.constexpr, out_dtype: tl.constexpr,
                                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pe = tl.program_id(2)
    m = tl.load(m_table + pe)
    n = tl.load(n_table + pe)
    mid = tl.load(mid_table + pe)

    stride_base = pe * 4
    gate_s0 = tl.load(gate_stride_table + stride_base + 2)
    gate_s1 = tl.load(gate_stride_table + stride_base + 3)
    lhs_s0 = tl.load(lhs_stride_table + stride_base + 2)
    lhs_s1 = tl.load(lhs_stride_table + stride_base + 3)
    up_s0 = tl.load(up_rhs_stride_table + stride_base + 2)
    up_s1 = tl.load(up_rhs_stride_table + stride_base + 3)
    down_s0 = tl.load(down_rhs_stride_table + stride_base + 2)
    down_s1 = tl.load(down_rhs_stride_table + stride_base + 3)
    out_s0 = tl.load(out_stride_table + stride_base + 2)
    out_s1 = tl.load(out_stride_table + stride_base + 3)

    gate_base = tl.load(gate_ptrs + pe)
    lhs_base = tl.load(lhs_ptrs + pe)
    up_base = tl.load(up_rhs_ptrs + pe)
    down_base = tl.load(down_rhs_ptrs + pe)
    out_base = tl.load(out_ptrs + pe)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = tl.arange(0, BLOCK_J)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for j0 in range(0, K_MID, BLOCK_J):
        j = j0 + offs_j
        up_acc = tl.zeros((BLOCK_M, BLOCK_J), tl.float32)

        for k0 in range(0, K_LHS, BLOCK_K):
            k = k0 + offs_k
            lhs_offsets = offs_m[:, None] * lhs_s0 + k[None, :] * lhs_s1
            up_offsets = k[:, None] * up_s0 + j[None, :] * up_s1
            lhs = _nncase_load_by_type(
                lhs_base, lhs_offsets, (offs_m[:, None] < m) & (k[None, :] < K_LHS), lhs_dtype)
            up_rhs = _nncase_load_by_type(
                up_base, up_offsets, (k[:, None] < K_LHS) & (j[None, :] < mid), up_rhs_dtype)
            up_acc += tl.dot(lhs, up_rhs)

        gate_offsets = offs_m[:, None] * gate_s0 + j[None, :] * gate_s1
        gate = _nncase_load_by_type(
            gate_base, gate_offsets, (offs_m[:, None] < m) & (j[None, :] < mid), gate_dtype).to(tl.float32)
        down_offsets = j[:, None] * down_s0 + offs_n[None, :] * down_s1
        down_rhs = _nncase_load_by_type(
            down_base, down_offsets, (j[:, None] < mid) & (offs_n[None, :] < n), down_rhs_dtype).to(tl.float32)
        acc += tl.dot((gate * up_acc).to(tl.float32), down_rhs)

    out_offsets = offs_m[:, None] * out_s0 + offs_n[None, :] * out_s1
    _nncase_store_by_type(out_base, out_offsets, acc, (offs_m[:, None] < m) & (offs_n[None, :] < n), out_dtype)
