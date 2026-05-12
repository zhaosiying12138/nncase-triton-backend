# op 7: device_func_26198

- Qwen3 layer note: RoPE cosine table: position_ids * inv_freq then cos
- nncase kind: `function`
- nncase arguments: `['buffer_20', 'const_21', 'dim_var', 'buffer_22', 'buffer_23']`
- generated Triton kernels: `_nncase_pe_binary_rank4_kernel, _nncase_pe_unary_rank4_kernel`
- op attrs: `{"function_name": "device_func_26198"}`

Internal non-memcopy nncase ops inside this generated function:

- `2: compute elementwise.mul attrs={"binary_op": "Mul", "lhs_padded_nums": [], "lhs_vectorized_axes": [], "rhs_padded_nums": [], "rhs_vectorized_axes": []}`
- `4: compute elementwise.cos attrs={"unary_op": "Cos"}`

Selected buffer descriptors:

- `arg0` `buffer_20` dtype=DataTypes.Float32 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 1}] dist=f32[sequence_length,1], (B,B), [p:20], Partial: False
- `arg1` `const_21` dtype=DataTypes.Float32 shape=[{'kind': 'fixed', 'value': 128}] dist=f32[128], (B), [p:20], Partial: False
- `arg3` `buffer_22` dtype=DataTypes.Float32 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[sequence_length,128], (B,B), [p:20], Partial: False
- `arg4` `buffer_23` dtype=DataTypes.Float32 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[sequence_length,128], (B,B), [p:20], Partial: False
