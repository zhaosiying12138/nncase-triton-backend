# op 25: device_func_26661

- Qwen3 layer note: MLP up projection + elementwise gate/up multiply + down projection
- nncase kind: `function`
- nncase arguments: `['const_47', 'buffer_53', 'buffer_50', 'const_54', 'dim_var', 'buffer_55']`
- generated Triton kernels: `_nncase_pe_matmul_mul_matmul_kernel`
- op attrs: `{"function_name": "device_func_26661"}`

Internal non-memcopy nncase ops inside this generated function:

- `3: matmul matmul attrs={"c_source_path": "", "func_name": "", "fused_reduce": false, "lhs_vectorized_axes": [], "rhs_vectorized_axes": [], "transpose_a": false, "transpose_b": false}`
- `5: compute elementwise.mul attrs={"binary_op": "Mul", "lhs_padded_nums": [], "lhs_vectorized_axes": [], "rhs_padded_nums": [], "rhs_vectorized_axes": []}`
- `7: matmul matmul attrs={"c_source_path": "", "func_name": "", "fused_reduce": false, "lhs_vectorized_axes": [], "rhs_vectorized_axes": [], "transpose_a": false, "transpose_b": false}`

Runtime note: for PE=20, the CUDA/Triton backend recognizes this whole function and emits `_nncase_pe_matmul_mul_matmul_kernel` directly on parent descriptors. This avoids the callee L0 tile shapes that were generated for a different fixed tiling factor.

Selected buffer descriptors:

- `arg0` `const_47` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 3072}, {'kind': 'fixed', 'value': 1024}] dist=f16[3072,1024], (B,B), [p:20], Partial: False
- `arg1` `buffer_53` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 3072}] dist=f16[sequence_length,3072], (B,B), [p:20], Partial: False
- `arg2` `buffer_50` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 1024}] dist=f16[sequence_length,1024], (B,B), [p:20], Partial: False
- `arg3` `const_54` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 1024}, {'kind': 'fixed', 'value': 3072}] dist=f16[1024,3072], (B,B), [p:20], Partial: False
- `arg5` `buffer_55` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 1024}] dist=f16[sequence_length,1024], (B,B), [p:20], Partial: False
