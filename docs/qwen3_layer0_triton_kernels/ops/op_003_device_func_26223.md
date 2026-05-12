# op 3: device_func_26223

- Qwen3 layer note: prefill/decode mask where + attention input RMSNorm + one 1024-wide attention projection; also preserves the residual tensor for the later add
- nncase kind: `function`
- nncase arguments: `['const_0', 'const_1', 'const_2', 'buffer_6', 'const_7', 'buffer_9', 'dim_var', 'buffer_10', 'buffer_11', 'buffer_12']`
- generated Triton kernels: `_nncase_pe_where_rank4_kernel, _nncase_pe_layer_norm_kernel, _nncase_pe_matmul_kernel`
- op attrs: `{"function_name": "device_func_26223"}`

Internal non-memcopy nncase ops inside this generated function:

- `4: compute elementwise.where attrs={}`
- `8: compute vectorized_layer_norm attrs={"axis": 1, "c_source_path": "", "epsilon": 1e-06, "func_name": "", "padded_nums": [], "use_mean": false, "vectorized_axes": []}`
- `11: matmul matmul attrs={"c_source_path": "", "func_name": "", "fused_reduce": false, "lhs_vectorized_axes": [], "rhs_vectorized_axes": [], "transpose_a": false, "transpose_b": false}`

Selected buffer descriptors:

- `arg0` `const_0` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 1024}, {'kind': 'fixed', 'value': 1024}] dist=f16[1024,1024], (B,B), [p:20], Partial: False
- `arg1` `const_1` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 1024}] dist=f16[1024], (B), [p:20], Partial: False
- `arg2` `const_2` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 1024}] dist=f16[1024], (B), [p:20], Partial: False
- `arg3` `buffer_6` dtype=DataTypes.Boolean shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 1}] dist=bool[sequence_length,1], (B,B), [p:20], Partial: False
- `arg4` `const_7` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 1}] dist=f16[1], (B), [p:20], Partial: False
- `arg5` `buffer_9` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 1024}] dist=f16[sequence_length,1024], (B,B), [p:20], Partial: False
