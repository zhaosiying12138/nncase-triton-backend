# op 10: device_func_26254

- Qwen3 layer note: prepare q for paged attention as [heads,head_dim,S]
- nncase kind: `function`
- nncase arguments: `['buffer_25', 'dim_var', 'buffer_26']`
- generated Triton kernels: `_nncase_pe_transpose_rank4_kernel`
- op attrs: `{"function_name": "device_func_26254"}`

Internal non-memcopy nncase ops inside this generated function:

- `1: compute elementwise.cast attrs={"cast_mode": "KDefault", "new_type": "DataTypes.Float16", "vectorize_axes": []}`
- `2: compute transpose attrs={"perm": [0, 2, 1]}`

Selected buffer descriptors:

- `arg0` `buffer_25` dtype=DataTypes.Float32 shape=[{'kind': 'fixed', 'value': 16}, {'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[16,sequence_length,128], (B,B,B), [p:20], Partial: False
- `arg2` `buffer_26` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 16}, {'kind': 'fixed', 'value': 128}, {'kind': 'dynamic', 'symbol': 'sequence_length'}] dist=f16[16,128,sequence_length], (B,B,B), [p:20], Partial: False
