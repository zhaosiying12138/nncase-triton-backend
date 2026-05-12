# op 20: device_func_26256

- Qwen3 layer note: o_proj: attention output [S,2048] -> hidden [S,1024]
- nncase kind: `function`
- nncase arguments: `['buffer_43', 'const_44', 'dim_var', 'buffer_45']`
- generated Triton kernels: `_nncase_pe_matmul_kernel`
- op attrs: `{"function_name": "device_func_26256"}`

Internal non-memcopy nncase ops inside this generated function:

- `3: matmul matmul attrs={"c_source_path": "", "func_name": "", "fused_reduce": false, "lhs_vectorized_axes": [], "rhs_vectorized_axes": [], "transpose_a": false, "transpose_b": false}`

Selected buffer descriptors:

- `arg0` `buffer_43` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 2048}] dist=f16[sequence_length,2048], (B,B), [p:20], Partial: False
- `arg1` `const_44` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 2048}, {'kind': 'fixed', 'value': 1024}] dist=f16[2048,1024], (B,B), [p:20], Partial: False
- `arg3` `buffer_45` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 1024}] dist=f16[sequence_length,1024], (B,B), [p:20], Partial: False
