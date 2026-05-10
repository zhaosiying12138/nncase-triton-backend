# CUDA Triton fused-kernel modes

`--fused-kernel={off,compute,compute-ccl}` is carried as a CUDA/NTT target option and emitted into the generated Triton module metadata as `fused_kernel`.

The persistent tile contract belongs to the CUDA/Triton baseline, not to a separate fused-kernel mode. Baseline admission defaults to `PE=16`, can be overridden by `NNCASE_CUDA_TILE_PE`, rejects mismatched PE counts, and rejects devices whose SM count cannot resident the requested PE count. Persistent kernels must use a single-dimensional `grid=(PE,)` and `pe = tl.program_id(0)`.

Modes:

- `off`: default. Uses the persistent tile baseline but disables whole-function compute fusion, so it remains the regression baseline for launch-sequence fusion.
- `compute`: enables CUDA target pass fusion for PE-local compute chains that are already backed by Triton-native helpers. Dense attention `QK^T -> scalar scale -> softmax -> AV` is recognized by an nncase pass and rewritten to an explicit `Fusion("cuda.flash_attention_*", "cuda", ...)`; codegen lowers that explicit Fusion name to the PE-local flash-attention helper for rank3/rank4 tensors with `head_dim <= 128`. Patterns that need sequence/head-dim cross-PE CCL are rejected in this mode. Older generated-Python whole-function matchers remain as a transition path for compute patterns that have not yet moved to pass-level Fusion rules. Whole-function matches that contain CCL launches are rejected in this mode.
- `compute-ccl`: enables compute fusion and CCL-aware fused paths. The first implemented path recognizes rank4 `Partial + non-Partial -> non-Partial` add in the generated native dispatcher and runs `_nncase_partial_add_rank4_kernel` as one persistent PE-grid Triton launch. The kernel reads partial GMEM pointer tables, reduces across `src_pe`, adds the PE-local residual, and writes the destination slice directly. Other tensor load/store and gather-reduce-scatter CCL shapes keep their explicit native CCL materialization boundaries.

Python test runners can set `NNCASE_CUDA_FUSED_KERNEL` to override the compiled metadata. The Qwen3 profiling helper also accepts `--fused-kernel` and always sets `NNCASE_CUDA_TILE_PE` from `--cuda-pe` for the persistent tile baseline.

Current CUDA/Triton strict mode still rejects unsupported native lowering instead of falling back to torch/CPU. For first-pass Qwen3 validation, use small token counts and `PE=16`.

Current implementation note: the mode/API plumbing, baseline admission check, and generated multi-PE native Triton launches now use the persistent `grid=(PE,)` helper. Current native CCL materialization and the partial-add fused helper write directly to the API-selected destination GMEM buffers: all-gather/scatter style paths copy between PE-visible GMEM pointer tables, partial-reduce paths reduce from GMEM partials into the destination slice, and partial-add additionally applies the PE-local add in the same kernel. Metadata therefore leaves `ccl_scratch_bytes` at zero for these paths. Runtime still carries an optional scratch pointer/size ABI for future protocols that need scratch/counter state, but that scratch/counter protocol is not required by the current native paths.

Attention note: dense flash attention and paged attention are not conflicting paths. Dense attention graphs are fused through the pass-level `cuda.flash_attention_*` Fusion rule. The Qwen3 autoregressive graph uses paged KV-cache layout, so its `paged_attention` op lowers to `_nncase_paged_flash_attention_rank3_kernel`, a prewritten Flash-style online-softmax Triton kernel that reads the current Head/Dim/Seq paged KV cache through owner/slot tables. The legacy paged-attention kernel remains in the generated source as bring-up/reference code, but the native paged-attention dispatcher launches the paged-flash kernel and tags launch metadata with `"paged_flash": True`.

The current Qwen3 metadata does not contain a partial-input `elementwise.add`, so `_nncase_partial_add_rank4_kernel` is covered by the source contract and a focused CUDA micro-test instead of by the Qwen3 end-to-end graph.
