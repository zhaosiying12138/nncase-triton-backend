# op 12: device_func_26232

- Qwen3 layer note: prepare value cache tensor as [kv_heads=8,head_dim,S]
- nncase kind: `function`
- nncase arguments: `['buffer_29', 'dim_var', 'buffer_30']`
- generated Triton kernels: `_nncase_pe_transpose_rank4_kernel`
- op attrs: `{"function_name": "device_func_26232"}`

Internal non-memcopy nncase ops inside this generated function:

- `1: compute transpose attrs={"perm": [1, 2, 0]}`

Selected buffer descriptors:

- `arg0` `buffer_29` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 8}, {'kind': 'fixed', 'value': 128}] dist=f16[sequence_length,8,128], (B,B,B), [p:20], Partial: False
- `arg2` `buffer_30` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 8}, {'kind': 'fixed', 'value': 128}, {'kind': 'dynamic', 'symbol': 'sequence_length'}] dist=f16[8,128,sequence_length], (B,B,B), [p:20], Partial: False
