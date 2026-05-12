# op 4: device_func_26206

- Qwen3 layer note: attention q_proj: hidden [S,1024] -> q [S,2048]
- nncase kind: `function`
- nncase arguments: `['buffer_12', 'const_13', 'dim_var', 'buffer_14']`
- generated Triton kernels: `_nncase_pe_matmul_kernel`
- op attrs: `{"function_name": "device_func_26206"}`

Internal non-memcopy nncase ops inside this generated function:

- `3: matmul matmul attrs={"c_source_path": "", "func_name": "", "fused_reduce": false, "lhs_vectorized_axes": [], "rhs_vectorized_axes": [], "transpose_a": false, "transpose_b": false}`

Selected buffer descriptors:

- `arg0` `buffer_12` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 1024}] dist=f16[sequence_length,1024], (B,B), [p:20], Partial: False
- `arg1` `const_13` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 1024}, {'kind': 'fixed', 'value': 2048}] dist=f16[1024,2048], (B,B), [p:20], Partial: False
- `arg3` `buffer_14` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 2048}] dist=f16[sequence_length,2048], (B,B), [p:20], Partial: False
