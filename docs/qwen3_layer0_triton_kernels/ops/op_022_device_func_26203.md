# op 22: device_func_26203

- Qwen3 layer note: post-attention RMSNorm
- nncase kind: `function`
- nncase arguments: `['buffer_46', 'const_48', 'const_49', 'dim_var', 'buffer_50']`
- generated Triton kernels: `_nncase_pe_layer_norm_kernel`
- op attrs: `{"function_name": "device_func_26203"}`

Internal non-memcopy nncase ops inside this generated function:

- `3: compute vectorized_layer_norm attrs={"axis": 1, "c_source_path": "", "epsilon": 1e-06, "func_name": "", "padded_nums": [], "use_mean": false, "vectorized_axes": []}`

Selected buffer descriptors:

- `arg0` `buffer_46` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 1024}] dist=f16[sequence_length,1024], (B,B), [p:20], Partial: False
- `arg1` `const_48` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 1024}] dist=f16[1024], (B), [p:20], Partial: False
- `arg2` `const_49` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 1024}] dist=f16[1024], (B), [p:20], Partial: False
- `arg4` `buffer_50` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 1024}] dist=f16[sequence_length,1024], (B,B), [p:20], Partial: False
