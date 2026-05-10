"""Annotated Triton kernels used by Qwen3-0.6B layer 0, shard-16 PoC.

This file is a learning copy extracted from the generated module:

    tests_output/test_qwen3_cuda_poc/cuda_admission/pe_16/CodeGen/cuda/triton_module.py

It is not imported by the runtime.  The comments explain how to read the
generated kernels, especially the simulated NUMA PE dimension.
"""

import triton
import triton.language as tl


# Dtype codes used by the generated runtime metadata.
# 0 bool, 2 int8, 3 int16, 4 int32, 5 int64, 6 uint8,
# 10 float16, 13 bfloat16, otherwise float32.


@triton.jit
def _nncase_load_by_type(base, offsets, mask, dtype_code: tl.constexpr):
    """Load from a raw byte pointer with the dtype selected by metadata."""
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


@triton.jit
def _nncase_store_by_type(base, offsets, values, mask, dtype_code: tl.constexpr):
    """Store to a raw byte pointer with the dtype selected by metadata."""
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


@triton.jit
def _nncase_rank4_coords(offsets, shape_table, pe):
    """Convert a flat PE-local element offset into rank-4 coordinates.

    The generated runtime represents rank-0..rank-4 tensors as rank-4 by
    padding missing dimensions with 1. shape_table is laid out as:

        [pe0_dim0, pe0_dim1, pe0_dim2, pe0_dim3,
         pe1_dim0, pe1_dim1, ...]
    """
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


@triton.jit
def _nncase_rank4_linear(i0, i1, i2, i3, stride_table, pe):
    """Convert rank-4 coordinates into a PE-local linear offset."""
    base = pe * 4
    s0 = tl.load(stride_table + base + 0)
    s1 = tl.load(stride_table + base + 1)
    s2 = tl.load(stride_table + base + 2)
    s3 = tl.load(stride_table + base + 3)
    return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3


@triton.jit
def _nncase_rank4_broadcast_linear(i0, i1, i2, i3, shape_table, stride_table, pe):
    """Like _nncase_rank4_linear, but supports broadcast dimensions.

    If a source dimension is size 1, every output coordinate maps to source
    coordinate 0 for that dimension.
    """
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


@triton.jit
def _nncase_pe_copy_rank4_kernel(src_ptrs, dst_ptrs, total_table, shape_table, src_stride_table, dst_stride_table,
                                 src_dtype: tl.constexpr, dst_dtype: tl.constexpr, BLOCK: tl.constexpr):
    """PE-local copy/cast kernel.

    Used in layer 0 by q/k norm+transpose paths after the transpose to realize
    the final cast/copy. Grid is (tiles, PE).
    """
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    i0, i1, i2, i3 = _nncase_rank4_coords(offsets, shape_table, pe)
    src_offsets = _nncase_rank4_linear(i0, i1, i2, i3, src_stride_table, pe)
    dst_offsets = _nncase_rank4_linear(i0, i1, i2, i3, dst_stride_table, pe)
    values = _nncase_load_by_type(tl.load(src_ptrs + pe), src_offsets, mask, src_dtype)
    _nncase_store_by_type(tl.load(dst_ptrs + pe), dst_offsets, values, mask, dst_dtype)


@triton.jit
def _nncase_pe_unary_rank4_kernel(src_ptrs, dst_ptrs, total_table, shape_table, src_stride_table, dst_stride_table,
                                  op_code: tl.constexpr, src_dtype: tl.constexpr, dst_dtype: tl.constexpr,
                                  beta: tl.constexpr, BLOCK: tl.constexpr):
    """PE-local unary kernel.

    Layer 0 uses this for RoPE cos/sin generation:
      op_code 1 -> cos
      op_code 2 -> sin
      op_code 3 -> swish
    """
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    i0, i1, i2, i3 = _nncase_rank4_coords(offsets, shape_table, pe)
    src_offsets = _nncase_rank4_linear(i0, i1, i2, i3, src_stride_table, pe)
    dst_offsets = _nncase_rank4_linear(i0, i1, i2, i3, dst_stride_table, pe)
    x = _nncase_load_by_type(tl.load(src_ptrs + pe), src_offsets, mask, src_dtype).to(tl.float32)
    y = x
    if op_code == 1:
        y = tl.cos(x)
    elif op_code == 2:
        y = tl.sin(x)
    elif op_code == 3:
        y = x * tl.sigmoid(x * beta)
    _nncase_store_by_type(tl.load(dst_ptrs + pe), dst_offsets, y, mask, dst_dtype)


@triton.jit
def _nncase_pe_binary_rank4_kernel(lhs_ptrs, rhs_ptrs, dst_ptrs, total_table, shape_table,
                                   lhs_shape_table, rhs_shape_table,
                                   lhs_stride_table, rhs_stride_table, dst_stride_table,
                                   op_code: tl.constexpr, lhs_dtype: tl.constexpr, rhs_dtype: tl.constexpr,
                                   dst_dtype: tl.constexpr, BLOCK: tl.constexpr):
    """PE-local binary kernel with broadcasting.

    Layer 0 uses this for:
      - RoPE angle = position_ids * inv_freq
      - residual adds

    op_code 0 -> add, 1 -> multiply, 2 -> equality.
    """
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    i0, i1, i2, i3 = _nncase_rank4_coords(offsets, shape_table, pe)
    lhs_offsets = _nncase_rank4_broadcast_linear(i0, i1, i2, i3, lhs_shape_table, lhs_stride_table, pe)
    rhs_offsets = _nncase_rank4_broadcast_linear(i0, i1, i2, i3, rhs_shape_table, rhs_stride_table, pe)
    dst_offsets = _nncase_rank4_linear(i0, i1, i2, i3, dst_stride_table, pe)
    left = _nncase_load_by_type(tl.load(lhs_ptrs + pe), lhs_offsets, mask, lhs_dtype)
    right = _nncase_load_by_type(tl.load(rhs_ptrs + pe), rhs_offsets, mask, rhs_dtype)
    y = left + right
    if op_code == 1:
        y = left * right
    elif op_code == 2:
        y = left == right
    _nncase_store_by_type(tl.load(dst_ptrs + pe), dst_offsets, y, mask, dst_dtype)


@triton.jit
def _nncase_pe_where_rank4_kernel(cond_ptrs, lhs_ptrs, rhs_ptrs, dst_ptrs, total_table, shape_table,
                                  cond_shape_table, lhs_shape_table, rhs_shape_table,
                                  cond_stride_table, lhs_stride_table, rhs_stride_table, dst_stride_table,
                                  cond_dtype: tl.constexpr, lhs_dtype: tl.constexpr, rhs_dtype: tl.constexpr,
                                  dst_dtype: tl.constexpr, BLOCK: tl.constexpr):
    """PE-local where/select.

    In ord5 this applies the validity/padding mask before the input RMSNorm and
    residual preservation. The kernel still obeys the PE-local pointer rule.
    """
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    i0, i1, i2, i3 = _nncase_rank4_coords(offsets, shape_table, pe)
    cond_offsets = _nncase_rank4_broadcast_linear(i0, i1, i2, i3, cond_shape_table, cond_stride_table, pe)
    lhs_offsets = _nncase_rank4_broadcast_linear(i0, i1, i2, i3, lhs_shape_table, lhs_stride_table, pe)
    rhs_offsets = _nncase_rank4_broadcast_linear(i0, i1, i2, i3, rhs_shape_table, rhs_stride_table, pe)
    dst_offsets = _nncase_rank4_linear(i0, i1, i2, i3, dst_stride_table, pe)
    pred = _nncase_load_by_type(tl.load(cond_ptrs + pe), cond_offsets, mask, cond_dtype) != 0
    left = _nncase_load_by_type(tl.load(lhs_ptrs + pe), lhs_offsets, mask, lhs_dtype)
    right = _nncase_load_by_type(tl.load(rhs_ptrs + pe), rhs_offsets, mask, rhs_dtype)
    _nncase_store_by_type(tl.load(dst_ptrs + pe), dst_offsets, tl.where(pred, left, right), mask, dst_dtype)


@triton.jit
def _nncase_pe_transpose_rank4_kernel(src_ptrs, dst_ptrs, total_table, dst_shape_table,
                                      src_stride_table, dst_stride_table,
                                      p0: tl.constexpr, p1: tl.constexpr, p2: tl.constexpr, p3: tl.constexpr,
                                      src_dtype: tl.constexpr, dst_dtype: tl.constexpr, BLOCK: tl.constexpr):
    """PE-local rank-4 transpose/cast.

    Used for q/k/v layout changes, for example:
      q [S,16,128] -> [16,S,128]
      K/V [8,S,128] -> [8,128,S]
    """
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    i0, i1, i2, i3 = _nncase_rank4_coords(offsets, dst_shape_table, pe)
    s0 = tl.zeros((BLOCK,), tl.int64)
    s1 = tl.zeros((BLOCK,), tl.int64)
    s2 = tl.zeros((BLOCK,), tl.int64)
    s3 = tl.zeros((BLOCK,), tl.int64)
    if p0 == 0:
        s0 = i0
    elif p0 == 1:
        s1 = i0
    elif p0 == 2:
        s2 = i0
    else:
        s3 = i0
    if p1 == 0:
        s0 = i1
    elif p1 == 1:
        s1 = i1
    elif p1 == 2:
        s2 = i1
    else:
        s3 = i1
    if p2 == 0:
        s0 = i2
    elif p2 == 1:
        s1 = i2
    elif p2 == 2:
        s2 = i2
    else:
        s3 = i2
    if p3 == 0:
        s0 = i3
    elif p3 == 1:
        s1 = i3
    elif p3 == 2:
        s2 = i3
    else:
        s3 = i3
    src_offsets = _nncase_rank4_linear(s0, s1, s2, s3, src_stride_table, pe)
    dst_offsets = _nncase_rank4_linear(i0, i1, i2, i3, dst_stride_table, pe)
    values = _nncase_load_by_type(tl.load(src_ptrs + pe), src_offsets, mask, src_dtype)
    _nncase_store_by_type(tl.load(dst_ptrs + pe), dst_offsets, values, mask, dst_dtype)


@triton.jit
def _nncase_pe_rope_rank4_kernel(x_ptrs, cos_ptrs, sin_ptrs, out_ptrs,
                                 total_table, out_shape_table,
                                 cos_shape_table, sin_shape_table,
                                 x_stride_table, cos_stride_table, sin_stride_table, out_stride_table,
                                 half_dim: tl.constexpr,
                                 x_dtype: tl.constexpr, cos_dtype: tl.constexpr, sin_dtype: tl.constexpr,
                                 out_dtype: tl.constexpr, BLOCK: tl.constexpr):
    """PE-local RoPE kernel.

    It rotates the last dimension in two halves:
      first half:  x*cos - partner*sin
      second half: x*cos + partner*sin
    Q uses CCL-broadcast cos/sin; K uses local cos/sin.
    """
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    i0, i1, i2, i3 = _nncase_rank4_coords(offsets, out_shape_table, pe)
    partner_dim = tl.where(i3 < half_dim, i3 + half_dim, i3 - half_dim)
    x_offsets = _nncase_rank4_linear(i0, i1, i2, i3, x_stride_table, pe)
    partner_offsets = _nncase_rank4_linear(i0, i1, i2, partner_dim, x_stride_table, pe)
    cos_offsets = _nncase_rank4_broadcast_linear(i0, i1, i2, i3, cos_shape_table, cos_stride_table, pe)
    sin_offsets = _nncase_rank4_broadcast_linear(i0, i1, i2, i3, sin_shape_table, sin_stride_table, pe)
    out_offsets = _nncase_rank4_linear(i0, i1, i2, i3, out_stride_table, pe)
    x = _nncase_load_by_type(tl.load(x_ptrs + pe), x_offsets, mask, x_dtype).to(tl.float32)
    partner = _nncase_load_by_type(tl.load(x_ptrs + pe), partner_offsets, mask, x_dtype).to(tl.float32)
    cos = _nncase_load_by_type(tl.load(cos_ptrs + pe), cos_offsets, mask, cos_dtype).to(tl.float32)
    sin = _nncase_load_by_type(tl.load(sin_ptrs + pe), sin_offsets, mask, sin_dtype).to(tl.float32)
    rotated = tl.where(i3 < half_dim, x * cos - partner * sin, x * cos + partner * sin)
    _nncase_store_by_type(tl.load(out_ptrs + pe), out_offsets, rotated, mask, out_dtype)


@triton.jit
def _nncase_pe_position_ids_kernel(dst_ptrs, total_table, start: tl.constexpr, local_offsets, dst_dtype: tl.constexpr, BLOCK: tl.constexpr):
    """Generate PE-local position ids for the token shard."""
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    base = tl.load(dst_ptrs + pe)
    local_offset = tl.load(local_offsets + pe)
    _nncase_store_by_type(base, offsets, start + local_offset + offsets, mask, dst_dtype)


@triton.jit
def _nncase_pe_update_kv_rank4_kernel(slots_ptrs, cache_ptrs,
                                      total_table, slots_shape_table, slots_stride_table, cache_stride_table,
                                      seq_offsets, start: tl.constexpr,
                                      slots_dtype: tl.constexpr, cache_dtype: tl.constexpr,
                                      BLOCK: tl.constexpr):
    """Write a PE-local K or V shard into that PE's KV-cache region."""
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(total_table + pe)
    mask = offsets < total
    i0, i1, i2, i3 = _nncase_rank4_coords(offsets, slots_shape_table, pe)
    src_offsets = _nncase_rank4_linear(i0, i1, i2, i3, slots_stride_table, pe)
    dst_i3 = i3 + start + tl.load(seq_offsets + pe)
    dst_offsets = _nncase_rank4_linear(i0, i1, i2, dst_i3, cache_stride_table, pe)
    values = _nncase_load_by_type(tl.load(slots_ptrs + pe), src_offsets, mask, slots_dtype)
    _nncase_store_by_type(tl.load(cache_ptrs + pe), dst_offsets, values, mask, cache_dtype)


@triton.jit
def _nncase_pe_matmul_kernel(lhs_ptrs, rhs_ptrs, out_ptrs,
                             m_table, n_table, K: tl.constexpr,
                             lhs_stride_table, rhs_stride_table, out_stride_table,
                             lhs_dtype: tl.constexpr, rhs_dtype: tl.constexpr, out_dtype: tl.constexpr,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """PE-local tiled matmul.

    Grid is (M tiles, N tiles, PE). The PE id selects one lhs/rhs/out pointer.
    For ord25 o_proj, K is only the local head shard width: 2048 / 16 = 128,
    and the output is Partial=True until the following CCL reduce-scatter.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pe = tl.program_id(2)
    m = tl.load(m_table + pe)
    n = tl.load(n_table + pe)
    stride_base = pe * 4
    lhs_s0 = tl.load(lhs_stride_table + stride_base + 2)
    lhs_s1 = tl.load(lhs_stride_table + stride_base + 3)
    rhs_s0 = tl.load(rhs_stride_table + stride_base + 2)
    rhs_s1 = tl.load(rhs_stride_table + stride_base + 3)
    out_s0 = tl.load(out_stride_table + stride_base + 2)
    out_s1 = tl.load(out_stride_table + stride_base + 3)
    lhs_base = tl.load(lhs_ptrs + pe)
    rhs_base = tl.load(rhs_ptrs + pe)
    out_base = tl.load(out_ptrs + pe)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a_offsets = offs_m[:, None] * lhs_s0 + k[None, :] * lhs_s1
        b_offsets = k[:, None] * rhs_s0 + offs_n[None, :] * rhs_s1
        a = _nncase_load_by_type(lhs_base, a_offsets, (offs_m[:, None] < m) & (k[None, :] < K), lhs_dtype)
        b = _nncase_load_by_type(rhs_base, b_offsets, (k[:, None] < K) & (offs_n[None, :] < n), rhs_dtype)
        acc += tl.dot(a, b)
    out_offsets = offs_m[:, None] * out_s0 + offs_n[None, :] * out_s1
    _nncase_store_by_type(out_base, out_offsets, acc, (offs_m[:, None] < m) & (offs_n[None, :] < n), out_dtype)


@triton.jit
def _nncase_pe_silu_mul_matmul_kernel(gate_ptrs, up_ptrs, rhs_ptrs, out_ptrs,
                                      m_table, n_table, K: tl.constexpr,
                                      gate_stride_table, up_stride_table, rhs_stride_table, out_stride_table,
                                      gate_dtype: tl.constexpr, up_dtype: tl.constexpr, rhs_dtype: tl.constexpr, out_dtype: tl.constexpr,
                                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Fused MLP down path: silu(gate) * up, then matmul with down_proj."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pe = tl.program_id(2)
    m = tl.load(m_table + pe)
    n = tl.load(n_table + pe)
    stride_base = pe * 4
    gate_s0 = tl.load(gate_stride_table + stride_base + 2)
    gate_s1 = tl.load(gate_stride_table + stride_base + 3)
    up_s0 = tl.load(up_stride_table + stride_base + 2)
    up_s1 = tl.load(up_stride_table + stride_base + 3)
    rhs_s0 = tl.load(rhs_stride_table + stride_base + 2)
    rhs_s1 = tl.load(rhs_stride_table + stride_base + 3)
    out_s0 = tl.load(out_stride_table + stride_base + 2)
    out_s1 = tl.load(out_stride_table + stride_base + 3)
    gate_base = tl.load(gate_ptrs + pe)
    up_base = tl.load(up_ptrs + pe)
    rhs_base = tl.load(rhs_ptrs + pe)
    out_base = tl.load(out_ptrs + pe)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        lhs_mask = (offs_m[:, None] < m) & (k[None, :] < K)
        gate_offsets = offs_m[:, None] * gate_s0 + k[None, :] * gate_s1
        up_offsets = offs_m[:, None] * up_s0 + k[None, :] * up_s1
        rhs_offsets = k[:, None] * rhs_s0 + offs_n[None, :] * rhs_s1
        gate = _nncase_load_by_type(gate_base, gate_offsets, lhs_mask, gate_dtype).to(tl.float32)
        up = _nncase_load_by_type(up_base, up_offsets, lhs_mask, up_dtype).to(tl.float32)
        rhs = _nncase_load_by_type(rhs_base, rhs_offsets, (k[:, None] < K) & (offs_n[None, :] < n), rhs_dtype).to(tl.float32)
        silu = gate / (1.0 + tl.exp(-gate))
        acc += tl.dot((silu * up).to(tl.float32), rhs)
    out_offsets = offs_m[:, None] * out_s0 + offs_n[None, :] * out_s1
    _nncase_store_by_type(out_base, out_offsets, acc, (offs_m[:, None] < m) & (offs_n[None, :] < n), out_dtype)


@triton.jit
def _nncase_pe_layer_norm_kernel(x_ptrs, scale_ptrs, bias_ptrs, out_ptrs,
                                 row_table, cols: tl.constexpr,
                                 eps: tl.constexpr, use_mean: tl.constexpr,
                                 x_dtype: tl.constexpr, scale_dtype: tl.constexpr,
                                 bias_dtype: tl.constexpr, out_dtype: tl.constexpr,
                                 BLOCK: tl.constexpr):
    """PE-local RMSNorm/layernorm over one row.

    Qwen3 RMSNorm uses use_mean=False. For [S,H] with dist=(S(0),B), each PE
    owns a few sequence rows and the full H=1024 columns, so no CCL is needed.
    """
    row = tl.program_id(0)
    pe = tl.program_id(1)
    rows = tl.load(row_table + pe)
    row_active = row < rows
    cols_offsets = tl.arange(0, BLOCK)
    mask = (cols_offsets < cols) & row_active
    x_base = tl.load(x_ptrs + pe)
    scale_base = tl.load(scale_ptrs + pe)
    bias_base = tl.load(bias_ptrs + pe)
    out_base = tl.load(out_ptrs + pe)
    values = _nncase_load_by_type(x_base, row * cols + cols_offsets, mask, x_dtype).to(tl.float32)
    if use_mean:
        mean = tl.sum(values, axis=0) / cols
        centered = tl.where(mask, values - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / cols
        norm = centered * tl.rsqrt(var + eps)
    else:
        var = tl.sum(tl.where(mask, values * values, 0.0), axis=0) / cols
        norm = values * tl.rsqrt(var + eps)
    s = _nncase_load_by_type(scale_base, cols_offsets, mask, scale_dtype).to(tl.float32)
    b = _nncase_load_by_type(bias_base, cols_offsets, mask, bias_dtype).to(tl.float32)
    _nncase_store_by_type(out_base, row * cols + cols_offsets, norm * s + b, mask, out_dtype)


@triton.jit
def _nncase_ccl_rank4_kernel(src_ptrs, dst_ptrs,
                             dst_total_table, dst_shape_table, dst_offset_table,
                             src_shape_table, src_offset_table,
                             src_stride_table, dst_stride_table,
                             g0: tl.constexpr, g1: tl.constexpr, g2: tl.constexpr, g3: tl.constexpr,
                             src_shard_axis: tl.constexpr, dst_shard_axis: tl.constexpr,
                             reduce_partial: tl.constexpr, pe_count: tl.constexpr,
                             src_dtype: tl.constexpr, dst_dtype: tl.constexpr,
                             BLOCK: tl.constexpr):
    """Rank-4 CCL/materialization kernel.

    This is the main place where reading another PE is legal.

    If reduce_partial=False, each output global coordinate finds the source PE
    that owns that coordinate and copies it.

    If reduce_partial=True, every source PE may contain a partial value for the
    same output coordinate. The kernel loops over all PEs and accumulates.
    """
    tile = tl.program_id(0)
    pe = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(dst_total_table + pe)
    mask = offsets < total
    li0, li1, li2, li3 = _nncase_rank4_coords(offsets, dst_shape_table, pe)
    off_base = pe * 4
    gi0 = li0 + tl.load(dst_offset_table + off_base + 0)
    gi1 = li1 + tl.load(dst_offset_table + off_base + 1)
    gi2 = li2 + tl.load(dst_offset_table + off_base + 2)
    gi3 = li3 + tl.load(dst_offset_table + off_base + 3)
    owner = pe + tl.zeros((BLOCK,), tl.int64)
    if src_shard_axis == 0:
        local_dim = tl.cdiv(g0, pe_count)
        owner = gi0 // local_dim
    elif src_shard_axis == 1:
        local_dim = tl.cdiv(g1, pe_count)
        owner = gi1 // local_dim
    elif src_shard_axis == 2:
        local_dim = tl.cdiv(g2, pe_count)
        owner = gi2 // local_dim
    elif src_shard_axis == 3:
        local_dim = tl.cdiv(g3, pe_count)
        owner = gi3 // local_dim
    if reduce_partial:
        acc = tl.zeros((BLOCK,), tl.float32)
        for src_pe in range(0, pe_count):
            src_base = src_pe * 4
            src_off0 = tl.load(src_offset_table + src_base + 0)
            src_off1 = tl.load(src_offset_table + src_base + 1)
            src_off2 = tl.load(src_offset_table + src_base + 2)
            src_off3 = tl.load(src_offset_table + src_base + 3)
            src_n0 = tl.load(src_shape_table + src_base + 0)
            src_n1 = tl.load(src_shape_table + src_base + 1)
            src_n2 = tl.load(src_shape_table + src_base + 2)
            src_n3 = tl.load(src_shape_table + src_base + 3)
            src_valid = (
                mask
                & (gi0 >= src_off0) & (gi0 < src_off0 + src_n0)
                & (gi1 >= src_off1) & (gi1 < src_off1 + src_n1)
                & (gi2 >= src_off2) & (gi2 < src_off2 + src_n2)
                & (gi3 >= src_off3) & (gi3 < src_off3 + src_n3)
            )
            src_offsets = _nncase_rank4_linear(
                gi0 - src_off0, gi1 - src_off1, gi2 - src_off2, gi3 - src_off3,
                src_stride_table, src_pe)
            acc += _nncase_load_by_type(tl.load(src_ptrs + src_pe), src_offsets, src_valid, src_dtype).to(tl.float32)
        values = acc
    else:
        safe_owner = tl.where(owner < 0, 0, tl.where(owner >= pe_count, pe_count - 1, owner))
        src_base = safe_owner * 4
        src_off0 = tl.load(src_offset_table + src_base + 0)
        src_off1 = tl.load(src_offset_table + src_base + 1)
        src_off2 = tl.load(src_offset_table + src_base + 2)
        src_off3 = tl.load(src_offset_table + src_base + 3)
        src_n0 = tl.load(src_shape_table + src_base + 0)
        src_n1 = tl.load(src_shape_table + src_base + 1)
        src_n2 = tl.load(src_shape_table + src_base + 2)
        src_n3 = tl.load(src_shape_table + src_base + 3)
        s0 = tl.load(src_stride_table + src_base + 0)
        s1 = tl.load(src_stride_table + src_base + 1)
        s2 = tl.load(src_stride_table + src_base + 2)
        s3 = tl.load(src_stride_table + src_base + 3)
        si0 = gi0 - src_off0
        si1 = gi1 - src_off1
        si2 = gi2 - src_off2
        si3 = gi3 - src_off3
        src_valid = (
            mask
            & (owner >= 0) & (owner < pe_count)
            & (gi0 >= src_off0) & (gi0 < src_off0 + src_n0)
            & (gi1 >= src_off1) & (gi1 < src_off1 + src_n1)
            & (gi2 >= src_off2) & (gi2 < src_off2 + src_n2)
            & (gi3 >= src_off3) & (gi3 < src_off3 + src_n3)
        )
        src_offsets = si0 * s0 + si1 * s1 + si2 * s2 + si3 * s3
        values = _nncase_load_by_type(tl.load(src_ptrs + safe_owner), src_offsets, src_valid, src_dtype)
    dst_offsets = _nncase_rank4_linear(li0, li1, li2, li3, dst_stride_table, pe)
    _nncase_store_by_type(tl.load(dst_ptrs + pe), dst_offsets, values, mask, dst_dtype)


@triton.jit
def _nncase_collective_paged_attention_rank3_kernel(q_ptrs, key_ptrs, value_ptrs, out_ptrs, scale_ptrs,
                                                    q_shape_table, q_offset_table,
                                                    q_stride_table, key_stride_table, value_stride_table, out_stride_table,
                                                    owner_table,
                                                    start: tl.constexpr, end: tl.constexpr, repeat: tl.constexpr,
                                                    q_dtype: tl.constexpr, key_dtype: tl.constexpr,
                                                    value_dtype: tl.constexpr, out_dtype: tl.constexpr,
                                                    HEAD_DIM: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    """Collective paged attention for layer 0.

    q is PE-local by Q head. K/V cache pages are PE-owned by sequence/cache
    position. owner_table[t] tells which PE owns cache time index t.
    """
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
        k_offsets = kv_head * k_s1[None, :] + d[:, None] * k_s2[None, :] + t[None, :] * k_s3[None, :]
        k_vals = _nncase_load_by_type(key_base[None, :], k_offsets, d_mask[:, None] & valid_t[None, :], key_dtype).to(tl.float32)
        scores += tl.sum(k_vals * q_vals[:, None], axis=0)
    scale_base = tl.load(scale_ptrs + pe)
    scale = tl.load(scale_base.to(tl.pointer_type(tl.float32)))
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
    v_offsets = kv_head * v_s1 + out_dim * v_s2 + t * v_s3
    v_vals = _nncase_load_by_type(value_base, v_offsets, valid_t & active, value_dtype).to(tl.float32)
    acc = tl.sum(probs * v_vals, axis=0)
    out_base = tl.load(out_ptrs + pe)
    out_offsets = q_head * out_s1 + out_dim * out_s2 + q_seq * out_s3
    _nncase_store_by_type(out_base, out_offsets, acc, active & (out_dim < HEAD_DIM), out_dtype)
