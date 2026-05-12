# op 8: device_func_26196

- Qwen3 layer note: RoPE sine table: sin on the same angle buffer
- nncase kind: `function`
- nncase arguments: `['buffer_23', 'dim_var', 'buffer_24']`
- generated Triton kernels: `_nncase_pe_unary_rank4_kernel`
- op attrs: `{"function_name": "device_func_26196"}`

Internal non-memcopy nncase ops inside this generated function:

- `1: compute elementwise.sin attrs={"unary_op": "Sin"}`

Selected buffer descriptors:

- `arg0` `buffer_23` dtype=DataTypes.Float32 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[sequence_length,128], (B,B), [p:20], Partial: False
- `arg2` `buffer_24` dtype=DataTypes.Float32 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[sequence_length,128], (B,B), [p:20], Partial: False
