# op 18: paged_attention

- Qwen3 layer note: paged attention over the simulated per-PE KV cache
- nncase kind: `collective`
- nncase arguments: `['buffer_26', 'kvCache', 'buffer_39', 'const_40', 'buffer_41']`
- generated Triton kernels: `_nncase_collective_paged_attention_rank3_kernel`
- op attrs: `{"hidden_size": 2048, "layer_id": 0, "layout": ["Head", "Dim", "Seq"]}`

Selected buffer descriptors:

- `arg0` `buffer_26` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 16}, {'kind': 'fixed', 'value': 128}, {'kind': 'dynamic', 'symbol': 'sequence_length'}] dist=f16[16,128,sequence_length], (B,B,B), [p:20], Partial: False
- `arg2` `buffer_39` dtype=DataTypes.UInt8 shape=[{'kind': 'fixed', 'value': 8404992}] dist=u8[8404992], (B), [p:20], Partial: False
- `arg3` `const_40` dtype=DataTypes.Float16 shape=[] dist=None
- `arg4` `buffer_41` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 16}, {'kind': 'fixed', 'value': 128}, {'kind': 'dynamic', 'symbol': 'sequence_length'}] dist=f16[16,128,sequence_length], (B,B,B), [p:20], Partial: False
- `extra` `buffer_39` dtype=DataTypes.UInt8 shape=[{'kind': 'fixed', 'value': 8404992}] dist=u8[8404992], (B), [p:20], Partial: False
- `output` `buffer_41` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 16}, {'kind': 'fixed', 'value': 128}, {'kind': 'dynamic', 'symbol': 'sequence_length'}] dist=f16[16,128,sequence_length], (B,B,B), [p:20], Partial: False
