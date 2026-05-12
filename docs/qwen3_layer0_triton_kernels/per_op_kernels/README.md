# Per-Op Triton Kernels for Qwen3 Layer 0

This folder expands decoder layer 0 ordinal-by-ordinal. Each `op_*.py` file is a study copy that contains:

- the Qwen3-level meaning of that generated nncase op,
- the nncase launch kind and generated argument names,
- the PE/NUMA interpretation for the op,
- the `@triton.jit` kernel body or bodies used by that op.

These files are intentionally not imported by runtime. The runnable generated module remains:

`tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py`

The main reading order is ordinal order:

- `op_003_device_func_26223.py`: prefill/decode mask where + attention input RMSNorm + one 1024-wide attention projection; also preserves the residual tensor for the later add
- `op_004_device_func_26206.py`: attention q_proj: hidden [S,1024] -> q [S,2048]
- `op_005_device_func_26251.py`: q_norm + reshape/transpose/cast into [q_heads=16,S,head_dim=128]
- `op_006_device_func_26194.py`: position ids from paged KV state
- `op_007_device_func_26198.py`: RoPE cosine table: position_ids * inv_freq then cos
- `op_008_device_func_26196.py`: RoPE sine table: sin on the same angle buffer
- `op_009_ro_pe.py`: apply rotary embedding to q
- `op_010_device_func_26254.py`: prepare q for paged attention as [heads,head_dim,S]
- `op_011_device_func_26204.py`: attention k/v side 1024-wide projection from normalized hidden
- `op_012_device_func_26232.py`: prepare value cache tensor as [kv_heads=8,head_dim,S]
- `op_013_device_func_26240.py`: k_norm + reshape/transpose/cast into [kv_heads=8,S,head_dim=128]
- `op_014_ro_pe.py`: apply rotary embedding to k
- `op_015_device_func_26243.py`: prepare key cache tensor as [kv_heads=8,head_dim,S]
- `op_016_update_paged_attention_kvcache.py`: write key shard into simulated NUMA KV cache according to slot_mapping owner/local_slot
- `op_017_update_paged_attention_kvcache.py`: write value shard into simulated NUMA KV cache according to slot_mapping owner/local_slot
- `op_018_paged_attention.py`: paged attention over the simulated per-PE KV cache
- `op_019_device_func_26255.py`: transpose attention output back to [S,q_heads,head_dim]
- `op_020_device_func_26256.py`: o_proj: attention output [S,2048] -> hidden [S,1024]
- `op_021_device_func_26257.py`: attention residual add
- `op_022_device_func_26203.py`: post-attention RMSNorm
- `op_023_device_func_26259.py`: MLP gate projection: hidden [S,1024] -> [S,3072]
- `op_024_device_func_26261.py`: SiLU activation for gate branch
- `op_025_device_func_26661.py`: fused MLP up projection + gate/up multiply + down projection
- `op_026_device_func_26264.py`: MLP residual add; this completes decoder layer 0

## Important Interpretation

For most files, the PE id is encoded in a Triton grid dimension and the kernel selects PE-local pointers from tables like `src_ptrs`, `lhs_ptrs`, `rhs_ptrs`, and `out_ptrs`. That is the NUMA simulation boundary: the kernel's normal data path operates on the GMEM region assigned to the current PE.

`op_018_paged_attention.py` is intentionally different. It is the collective-style attention op and reads owner/slot tables so it can model cross-PE KV-cache access using a Triton kernel instead of a CPU or Torch fallback.
