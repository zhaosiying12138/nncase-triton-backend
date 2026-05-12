# op 15: device_func_26243

- Qwen3 layer note: prepare key cache tensor as [kv_heads=8,head_dim,S]
- nncase kind: `function`
- nncase arguments: `['buffer_35', 'dim_var', 'buffer_36']`
- generated Triton kernels: `_nncase_pe_transpose_rank4_kernel`
- op attrs: `{"function_name": "device_func_26243"}`

Internal non-memcopy nncase ops inside this generated function:

- `1: compute elementwise.cast attrs={"cast_mode": "KDefault", "new_type": "DataTypes.Float16", "vectorize_axes": []}`
- `2: compute transpose attrs={"perm": [0, 2, 1]}`

Selected buffer descriptors:

- `arg0` `buffer_35` dtype=DataTypes.Float32 shape=[{'kind': 'fixed', 'value': 8}, {'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[8,sequence_length,128], (B,B,B), [p:20], Partial: False
- `arg2` `buffer_36` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 8}, {'kind': 'fixed', 'value': 128}, {'kind': 'dynamic', 'symbol': 'sequence_length'}] dist=f16[8,128,sequence_length], (B,B,B), [p:20], Partial: False
