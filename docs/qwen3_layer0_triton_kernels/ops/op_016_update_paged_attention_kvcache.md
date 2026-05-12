# op 16: update_paged_attention_kvcache

- Qwen3 layer note: write key shard into simulated NUMA KV cache according to slot_mapping owner/local_slot
- nncase kind: `compute`
- nncase arguments: `['buffer_36', 'kvCache']`
- generated Triton kernels: `_nncase_pe_update_kv_rank4_kernel`
- op attrs: `{"cache_kind": "Key", "layer_id": 0, "layout": ["Head", "Dim", "Seq"]}`

Selected buffer descriptors:

- `arg0` `buffer_36` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 8}, {'kind': 'fixed', 'value': 128}, {'kind': 'dynamic', 'symbol': 'sequence_length'}] dist=f16[8,128,sequence_length], (B,B,B), [p:20], Partial: False
- `slots` `buffer_36` dtype=DataTypes.Float16 shape=[{'kind': 'fixed', 'value': 8}, {'kind': 'fixed', 'value': 128}, {'kind': 'dynamic', 'symbol': 'sequence_length'}] dist=f16[8,128,sequence_length], (B,B,B), [p:20], Partial: False
