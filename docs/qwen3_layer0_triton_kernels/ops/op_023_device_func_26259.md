# op 23: device_func_26259

- Qwen3 layer note: MLP gate projection: hidden [S,1024] -> [S,3072]
- nncase kind: `function`
- nncase arguments: `['buffer_50', 'const_51', 'dim_var', 'buffer_52']`
- generated Triton kernels: `_nncase_pe_matmul_kernel`
- op attrs: `{"function_name": "device_func_26259"}`

Internal non-memcopy nncase ops inside this generated function:

- `3: matmul matmul attrs={"c_source_path": "", "func_name": "", "fused_reduce": false, "lhs_vectorized_axes": [], "rhs_vectorized_axes": [], "transpose_a": false, "transpose_b": false}`

Selected buffer descriptors:

- `arg0` `buffer_50` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 1024}] dist=f16[sequence_length,1024], (B,B), [p:20], Partial: False
- `arg1` `const_51` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 1024}, {'kind': 'fixed', 'value': 3072}] dist=f16[1024,3072], (B,B), [p:20], Partial: False
- `arg3` `buffer_52` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 3072}] dist=f16[sequence_length,3072], (B,B), [p:20], Partial: False
