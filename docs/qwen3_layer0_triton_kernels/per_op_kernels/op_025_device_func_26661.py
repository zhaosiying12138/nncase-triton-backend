# Qwen3 decoder layer 0, ordinal 25: device_func_26661
# Source module: tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py
# PE count captured in this export: 20
#
# Qwen3 meaning:
#   MLP up projection + elementwise gate/up multiply + down projection.
#
# nncase device function shape:
#   gate = silu(hidden @ gate_weight) is produced by ordinal 24.
#   up = hidden @ up_weight is produced inside this ordinal.
#   out = (gate * up) @ down_weight is produced inside this ordinal.
#
# Why this file uses one fused kernel:
#   The generated callee still contains historical L0 buffers sized like 3072/32.
#   For PE=20 that L0 shape is not a valid NUMA partition. The CUDA/Triton
#   backend therefore recognizes the whole device function and launches this
#   PE-grid Triton kernel directly on the parent descriptors.
#
# NUMA rule:
#   program_id(2) is the PE id. Each program loads ptrs[pe] for gate/lhs/up
#   weight/down weight/out, so normal dataflow remains inside one simulated
#   NUMA domain. There is no CPU or Torch fallback here.

import triton
import triton.language as tl

# Study note:
#   These helpers are defined in the generated runtime module and in
#   docs/qwen3_layer0_triton_kernels/kernels/. They are intentionally not
#   duplicated here so the op file stays focused on the generated kernel body.
#   _nncase_load_by_type(ptr, offsets, mask, dtype_code)
#   _nncase_store_by_type(ptr, offsets, values, mask, dtype_code)


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

    # Per-PE logical sizes. For Qwen3-0.6B layer 0 under PE=20 this is
    # normally M=S_local, N=1024, K_LHS=1024, K_MID=3072.
    m = tl.load(m_table + pe)
    n = tl.load(n_table + pe)
    mid = tl.load(mid_table + pe)

    # Rank-4 descriptor stride tables are flattened as 4 entries per PE.
    # The matrices are rank-2, so axes 2/3 are the row/column strides.
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

    # NUMA boundary: all bases come from pointer-table slot `pe`.
    gate_base = tl.load(gate_ptrs + pe)
    lhs_base = tl.load(lhs_ptrs + pe)
    up_base = tl.load(up_rhs_ptrs + pe)
    down_base = tl.load(down_rhs_ptrs + pe)
    out_base = tl.load(out_ptrs + pe)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = tl.arange(0, BLOCK_J)
    offs_k = tl.arange(0, BLOCK_K)

    # Outer accumulation computes out[:, n] over the full MLP intermediate
    # dimension. For each intermediate tile j, we first compute:
    #   up[:, j] = lhs[:, :] @ up_rhs[:, j]
    # then multiply by the already-silu'd gate[:, j], then dot with down_rhs.
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
