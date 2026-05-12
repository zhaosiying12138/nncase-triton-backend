# Qwen3 Decoder Layer 0 Op Sequence

Source: `tests_output/test_qwen3_cuda_poc/cuda_admission/pe_20/CodeGen/cuda/triton_module.py`

Layer boundary used here: `main_segment_1_prim` ordinals 3-26. Ordinals 0-2 are input tensor load, input-id condition, and embedding gather, so they are not counted as decoder layer 0.

| Ordinal | Qwen3-level meaning | nncase kind | generated launch | inner nncase ops | Triton kernels |
|---:|---|---|---|---|---|
| 3 | prefill/decode mask where + attention input RMSNorm + one 1024-wide attention projection; also preserves the residual tensor for the later add | function | device_func_26223 | elementwise.where + vectorized_layer_norm + matmul | _nncase_pe_where_rank4_kernel, _nncase_pe_layer_norm_kernel, _nncase_pe_matmul_kernel |
| 4 | attention q_proj: hidden [S,1024] -> q [S,2048] | function | device_func_26206 | matmul | _nncase_pe_matmul_kernel |
| 5 | q_norm + reshape/transpose/cast into [q_heads=16,S,head_dim=128] | function | device_func_26251 | vectorized_layer_norm + transpose + elementwise.cast | _nncase_pe_layer_norm_transpose_rank4_kernel |
| 6 | position ids from paged KV state | function | device_func_26194 | get_position_ids | _nncase_pe_position_ids_kernel |
| 7 | RoPE cosine table: position_ids * inv_freq then cos | function | device_func_26198 | elementwise.mul + elementwise.cos | _nncase_pe_binary_rank4_kernel, _nncase_pe_unary_rank4_kernel |
| 8 | RoPE sine table: sin on the same angle buffer | function | device_func_26196 | elementwise.sin | _nncase_pe_unary_rank4_kernel |
| 9 | apply rotary embedding to q | compute | ro_pe | ro_pe | _nncase_pe_rope_rank4_kernel |
| 10 | prepare q for paged attention as [heads,head_dim,S] | function | device_func_26254 | elementwise.cast + transpose | _nncase_pe_transpose_rank4_kernel |
| 11 | attention k/v side 1024-wide projection from normalized hidden | function | device_func_26204 | matmul | _nncase_pe_matmul_kernel |
| 12 | prepare value cache tensor as [kv_heads=8,head_dim,S] | function | device_func_26232 | transpose | _nncase_pe_transpose_rank4_kernel |
| 13 | k_norm + reshape/transpose/cast into [kv_heads=8,S,head_dim=128] | function | device_func_26240 | vectorized_layer_norm + transpose + elementwise.cast | _nncase_pe_layer_norm_transpose_rank4_kernel |
| 14 | apply rotary embedding to k | compute | ro_pe | ro_pe | _nncase_pe_rope_rank4_kernel |
| 15 | prepare key cache tensor as [kv_heads=8,head_dim,S] | function | device_func_26243 | elementwise.cast + transpose | _nncase_pe_transpose_rank4_kernel |
| 16 | write key shard into simulated NUMA KV cache according to slot_mapping owner/local_slot | compute | update_paged_attention_kvcache | update_paged_attention_kvcache | _nncase_pe_update_kv_rank4_kernel |
| 17 | write value shard into simulated NUMA KV cache according to slot_mapping owner/local_slot | compute | update_paged_attention_kvcache | update_paged_attention_kvcache | _nncase_pe_update_kv_rank4_kernel |
| 18 | paged attention over the simulated per-PE KV cache | collective | paged_attention | paged_attention | _nncase_collective_paged_attention_rank3_kernel |
| 19 | transpose attention output back to [S,q_heads,head_dim] | function | device_func_26255 | transpose | _nncase_pe_transpose_rank4_kernel |
| 20 | o_proj: attention output [S,2048] -> hidden [S,1024] | function | device_func_26256 | matmul | _nncase_pe_matmul_kernel |
| 21 | attention residual add | function | device_func_26257 | elementwise.add | _nncase_pe_binary_rank4_kernel |
| 22 | post-attention RMSNorm | function | device_func_26203 | vectorized_layer_norm | _nncase_pe_layer_norm_kernel |
| 23 | MLP gate projection: hidden [S,1024] -> [S,3072] | function | device_func_26259 | matmul | _nncase_pe_matmul_kernel |
| 24 | SiLU activation for gate branch | function | device_func_26261 | swish | _nncase_pe_unary_rank4_kernel |
| 25 | MLP up projection + elementwise gate/up multiply + down projection | function | device_func_26661 | matmul + elementwise.mul + matmul | _nncase_pe_matmul_mul_matmul_kernel |
| 26 | MLP residual add; this completes decoder layer 0 | function | device_func_26264 | elementwise.add | _nncase_pe_binary_rank4_kernel |
