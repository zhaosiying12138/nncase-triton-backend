# op 13: device_func_26240

- Qwen3 layer note: k_norm + reshape/transpose/cast into [kv_heads=8,S,head_dim=128]
- nncase kind: `function`
- nncase arguments: `['buffer_31', 'const_32', 'const_33', 'dim_var', 'buffer_34']`
- generated Triton kernels: `_nncase_pe_layer_norm_transpose_rank4_kernel`
- op attrs: `{"function_name": "device_func_26240"}`

Internal non-memcopy nncase ops inside this generated function:

- `3: compute vectorized_layer_norm attrs={"axis": 2, "c_source_path": "", "epsilon": 1e-06, "func_name": "", "padded_nums": [], "use_mean": false, "vectorized_axes": []}`
- `4: compute transpose attrs={"perm": [1, 0, 2]}`
- `5: compute elementwise.cast attrs={"cast_mode": "KDefault", "new_type": "DataTypes.Float32", "vectorize_axes": []}`

Selected buffer descriptors:

- `arg0` `buffer_31` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 8}, {'kind': 'fixed', 'value': 128}] dist=f16[sequence_length,8,128], (B,B,B), [p:20], Partial: False
- `arg1` `const_32` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 128}] dist=f16[128], (B), [p:20], Partial: False
- `arg2` `const_33` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 128}] dist=f16[128], (B), [p:20], Partial: False
- `arg4` `buffer_34` dtype=DataTypes.Float32 shape=[{'kind': 'fixed', 'value': 8}, {'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[8,sequence_length,128], (B,B,B), [p:20], Partial: False
