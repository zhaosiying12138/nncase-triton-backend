# op 21: device_func_26257

- Qwen3 layer note: attention residual add
- nncase kind: `function`
- nncase arguments: `['buffer_11', 'buffer_45', 'dim_var', 'buffer_46']`
- generated Triton kernels: `_nncase_pe_binary_rank4_kernel`
- op attrs: `{"function_name": "device_func_26257"}`

Internal non-memcopy nncase ops inside this generated function:

- `2: compute elementwise.add attrs={"binary_op": "Add", "lhs_padded_nums": [], "lhs_vectorized_axes": [], "rhs_padded_nums": [], "rhs_vectorized_axes": []}`

Selected buffer descriptors:

- `arg0` `buffer_11` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 1024}] dist=f16[sequence_length,1024], (B,B), [p:20], Partial: False
- `arg1` `buffer_45` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 1024}] dist=f16[sequence_length,1024], (B,B), [p:20], Partial: False
- `arg3` `buffer_46` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 1024}] dist=f16[sequence_length,1024], (B,B), [p:20], Partial: False
