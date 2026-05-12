# op 6: device_func_26194

- Qwen3 layer note: position ids from paged KV state
- nncase kind: `function`
- nncase arguments: `['kvCache', 'dim_var', 'buffer_19']`
- generated Triton kernels: `_nncase_pe_position_ids_kernel`
- op attrs: `{"function_name": "device_func_26194"}`

Internal non-memcopy nncase ops inside this generated function:

- `1: compute get_position_ids attrs={}`

Selected buffer descriptors:

- `arg2` `buffer_19` dtype=DataTypes.Float32 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}] dist=f32[sequence_length], (B), [p:20], Partial: False
