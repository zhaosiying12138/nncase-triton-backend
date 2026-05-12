# op 19: device_func_26255

- Qwen3 layer note: transpose attention output back to [S,q_heads,head_dim]
- nncase kind: `function`
- nncase arguments: `['buffer_41', 'dim_var', 'buffer_42']`
- generated Triton kernels: `_nncase_pe_transpose_rank4_kernel`
- op attrs: `{"function_name": "device_func_26255"}`

Internal non-memcopy nncase ops inside this generated function:

- `1: compute transpose attrs={"perm": [2, 0, 1]}`

Selected buffer descriptors:

- `arg0` `buffer_41` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 16}, {'kind': 'fixed', 'value': 128}, {'kind': 'dynamic', 'symbol': 'sequence_length'}] dist=f16[16,128,sequence_length], (B,B,B), [p:20], Partial: False
- `arg2` `buffer_42` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 16}, {'kind': 'fixed', 'value': 128}] dist=f16[sequence_length,16,128], (B,B,B), [p:20], Partial: False
