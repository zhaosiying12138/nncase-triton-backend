# op 24: device_func_26261

- Qwen3 layer note: SiLU activation for gate branch
- nncase kind: `function`
- nncase arguments: `['buffer_52', 'dim_var', 'buffer_53']`
- generated Triton kernels: `_nncase_pe_unary_rank4_kernel`
- op attrs: `{"function_name": "device_func_26261"}`

Internal non-memcopy nncase ops inside this generated function:

- `1: compute swish attrs={"beta": 1}`

Selected buffer descriptors:

- `arg0` `buffer_52` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 3072}] dist=f16[sequence_length,3072], (B,B), [p:20], Partial: False
- `arg2` `buffer_53` dtype=DataTypes.Float16 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 3072}] dist=f16[sequence_length,3072], (B,B), [p:20], Partial: False
