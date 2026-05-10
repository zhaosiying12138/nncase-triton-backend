# Layer 0 Kernel Launch Map

Layer 0 is `main_segment_1_prim` ordinals `5..32` in
`tests_output/qwen3_triton_3tok_verbose.log`.

| Ordinal | Official Qwen3 role | Generated Triton kernel(s) |
| --- | --- | --- |
| 5 | mask/select, `input_layernorm`, value projection setup | `_nncase_pe_where_rank4_kernel`, `_nncase_pe_layer_norm_kernel`, `_nncase_pe_matmul_kernel` |
| 6 | CCL materialize/broadcast `n0` for Q path | `_nncase_ccl_rank4_kernel` |
| 7 | `self_attn.q_proj` | `_nncase_pe_matmul_kernel` |
| 8 | `self_attn.q_norm`, transpose/cast | `_nncase_pe_layer_norm_kernel`, `_nncase_pe_transpose_rank4_kernel`, `_nncase_pe_copy_rank4_kernel` |
| 9 | position ids | `_nncase_pe_position_ids_kernel` |
| 10 | RoPE angle and cos | `_nncase_pe_binary_rank4_kernel`, `_nncase_pe_unary_rank4_kernel` |
| 11 | CCL broadcast cos for Q RoPE | `_nncase_ccl_rank4_kernel` |
| 12 | RoPE sin | `_nncase_pe_unary_rank4_kernel` |
| 13 | CCL broadcast sin for Q RoPE | `_nncase_ccl_rank4_kernel` |
| 14 | Q RoPE | `_nncase_pe_rope_rank4_kernel` |
| 15 | Q transpose to attention layout | `_nncase_pe_transpose_rank4_kernel` |
| 16 | V transpose/cast to cache layout | `_nncase_pe_transpose_rank4_kernel` |
| 17 | `self_attn.k_proj` | `_nncase_pe_matmul_kernel` |
| 18 | `self_attn.k_norm`, transpose/cast | `_nncase_pe_layer_norm_kernel`, `_nncase_pe_transpose_rank4_kernel`, `_nncase_pe_copy_rank4_kernel` |
| 19 | K RoPE | `_nncase_pe_rope_rank4_kernel` |
| 20 | K transpose to cache layout | `_nncase_pe_transpose_rank4_kernel` |
| 21 | update K cache | `_nncase_pe_update_kv_rank4_kernel` |
| 22 | update V cache | `_nncase_pe_update_kv_rank4_kernel` |
| 23 | collective paged attention | `_nncase_collective_paged_attention_rank3_kernel` |
| 24 | attention output transpose/cast | `_nncase_pe_transpose_rank4_kernel` |
| 25 | `self_attn.o_proj` local partial matmul | `_nncase_pe_matmul_kernel` |
| 26 | CCL reduce-scatter `o_proj` partials | `_nncase_ccl_rank4_kernel` |
| 27 | first residual add | `_nncase_pe_binary_rank4_kernel` |
| 28 | `post_attention_layernorm` | `_nncase_pe_layer_norm_kernel` |
| 29 | `mlp.gate_proj` | `_nncase_pe_matmul_kernel` |
| 30 | `mlp.up_proj` | `_nncase_pe_matmul_kernel` |
| 31 | `silu(gate) * up` and `mlp.down_proj` | `_nncase_pe_silu_mul_matmul_kernel` |
| 32 | second residual add | `_nncase_pe_binary_rank4_kernel` |

Unique kernels in this layer:

- `_nncase_pe_copy_rank4_kernel`
- `_nncase_pe_unary_rank4_kernel`
- `_nncase_pe_binary_rank4_kernel`
- `_nncase_pe_where_rank4_kernel`
- `_nncase_pe_transpose_rank4_kernel`
- `_nncase_pe_rope_rank4_kernel`
- `_nncase_pe_position_ids_kernel`
- `_nncase_pe_update_kv_rank4_kernel`
- `_nncase_pe_matmul_kernel`
- `_nncase_pe_silu_mul_matmul_kernel`
- `_nncase_pe_layer_norm_kernel`
- `_nncase_ccl_rank4_kernel`
- `_nncase_collective_paged_attention_rank3_kernel`
