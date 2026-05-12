# op 9: ro_pe

- Qwen3 layer note: apply rotary embedding to q
- nncase kind: `compute`
- nncase arguments: `['buffer_18', 'buffer_22', 'buffer_24', 'buffer_25']`
- generated Triton kernels: `_nncase_pe_rope_rank4_kernel`

Selected buffer descriptors:

- `arg0` `buffer_18` dtype=DataTypes.Float32 shape=[{'kind': 'fixed', 'value': 16}, {'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[16,sequence_length,128], (B,B,B), [p:20], Partial: False
- `arg1` `buffer_22` dtype=DataTypes.Float32 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[sequence_length,128], (B,B), [p:20], Partial: False
- `arg2` `buffer_24` dtype=DataTypes.Float32 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[sequence_length,128], (B,B), [p:20], Partial: False
- `arg3` `buffer_25` dtype=DataTypes.Float32 shape=[{'kind': 'fixed', 'value': 16}, {'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[16,sequence_length,128], (B,B,B), [p:20], Partial: False
- `cos` `buffer_22` dtype=DataTypes.Float32 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[sequence_length,128], (B,B), [p:20], Partial: False
- `input` `buffer_18` dtype=DataTypes.Float32 shape=[{'kind': 'fixed', 'value': 16}, {'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[16,sequence_length,128], (B,B,B), [p:20], Partial: False
