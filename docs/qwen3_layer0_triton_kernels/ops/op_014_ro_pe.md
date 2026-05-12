# op 14: ro_pe

- Qwen3 layer note: apply rotary embedding to k
- nncase kind: `compute`
- nncase arguments: `['buffer_34', 'buffer_22', 'buffer_24', 'buffer_35']`
- generated Triton kernels: `_nncase_pe_rope_rank4_kernel`

Selected buffer descriptors:

- `arg0` `buffer_34` dtype=DataTypes.Float32 shape=[{'kind': 'fixed', 'value': 8}, {'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[8,sequence_length,128], (B,B,B), [p:20], Partial: False
- `arg1` `buffer_22` dtype=DataTypes.Float32 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[sequence_length,128], (B,B), [p:20], Partial: False
- `arg2` `buffer_24` dtype=DataTypes.Float32 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[sequence_length,128], (B,B), [p:20], Partial: False
- `arg3` `buffer_35` dtype=DataTypes.Float32 shape=[{'kind': 'fixed', 'value': 8}, {'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[8,sequence_length,128], (B,B,B), [p:20], Partial: False
- `cos` `buffer_22` dtype=DataTypes.Float32 shape=[{'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[sequence_length,128], (B,B), [p:20], Partial: False
- `input` `buffer_34` dtype=DataTypes.Float32 shape=[{'kind': 'fixed', 'value': 8}, {'kind': 'dynamic', 'symbol': 'sequence_length'}, {'kind': 'fixed', 'value': 128}] dist=f32[8,sequence_length,128], (B,B,B), [p:20], Partial: False
