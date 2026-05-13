# Qwen3 CUDA AutoDistributed Gmem Shard Constraint

This report records the default no-cap vs default finite-cap comparison for Qwen3-0.6B CUDA PE=16 with `NNCASE_CUDA_FUSED_KERNEL=off`.
The evidence is generated from complete Qwen3 runs and from AutoDistributed `Solve.txt` plus CUDA `cuda_meta.json`; runtime allocator peak and rdata dedup are not used as the primary proof.

## Source Artifacts

- default no-cap: `tests_output/qwen3_off_default_nocap_final`
  - Solve: `tests_output/qwen3_off_default_nocap_final/cuda_admission/pe_16/05_AutoDistributedPass/0_AutoDistributed_cuda/main/Costs/Solve.txt`
  - Metadata: `tests_output/qwen3_off_default_nocap_final/cuda_admission/pe_16/CodeGen/cuda/cuda_meta.json`
  - Token compare: `tests_output/qwen3_off_default_nocap_final/token_compare.txt`
- default cap 589824: `tests_output/qwen3_off_default_cap_589824_final`
  - Solve: `tests_output/qwen3_off_default_cap_589824_final/cuda_admission/pe_16/05_AutoDistributedPass/0_AutoDistributed_cuda/main/Costs/Solve.txt`
  - Metadata: `tests_output/qwen3_off_default_cap_589824_final/cuda_admission/pe_16/CodeGen/cuda/cuda_meta.json`
  - Token compare: `tests_output/qwen3_off_default_cap_589824_final/token_compare.txt`

## Primary Result

| Run | CUDA PE gmem cap | Solver status | CUDA op-step constraints | picked live tensor gmem/PE | constrained node peak | resident step footprint (secondary) | token ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| default no-cap | `Disabled` | `Optimal` | `0` | `8437760` (8.05 MiB) | `8404992` (8.02 MiB) | `311296512` (296.88 MiB) | `1.0000` |
| default cap | `589824` | `Optimal` | `8946` | `589824` (0.56 MiB) | `525312` (0.50 MiB) | `19456512` (18.56 MiB) | `1.0000` |

The picked live tensor gmem/PE drops from `8437760` bytes to `589824` bytes, a `14.31x` reduction. The resident/static footprint is reported only as secondary context (`311296512` to `19456512` bytes); it is not used as proof. The primary evidence is the AutoDistributed picked shard/live-tensor footprint, not rdata dedup or runtime allocator behavior.

## Prefill Layer0 Shard Change

The comparison uses `main_segment_1_prim` ord0..34, the existing prefill layer0 slice. Output distributed types differ at these ordinals:

`0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 33, 34`

- Detailed table: [`qwen3-layer0-fused-modes/qwen3-gmem-cap-layer0.md`](qwen3-layer0-fused-modes/qwen3-gmem-cap-layer0.md)
- Graph: [`qwen3-layer0-fused-modes/qwen3-gmem-cap-layer0.svg`](qwen3-layer0-fused-modes/qwen3-gmem-cap-layer0.svg)

<img src="qwen3-layer0-fused-modes/qwen3-gmem-cap-layer0.svg" alt="Qwen3 layer0 no-cap vs cap shard comparison" style="width: 100%; height: auto;">

Early layer0 examples:

- ord0 changes `input_ids` from broadcast local load to sequence-sharded load.
- ord2..4 change the embedding/gather path by adding a sequence-sharded tensor load and selecting a sequence-sharded gather/reshard path under the cap.
- ord5..7 keep the layer0 hidden-state path under the cap before later resharding for matmul-compatible layouts.

## Verification Commands

Focused AutoDistributed tests:

```bash
env DOTNET_ROOT=$HOME/.dotnet/zsy-nncase-dotnet8 PATH="$HOME/.dotnet/zsy-nncase-dotnet8:$PATH" LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
  dotnet test src/Nncase.Tests/Nncase.Tests.csproj -c Release --no-restore \
  --filter "FullyQualifiedName~UnitTestQwenEmbeddingShardSearch|FullyQualifiedName~UnitTestCudaAutoDistMemoryConstraint" \
  --logger "console;verbosity=minimal"
```

Compiler publish used by the Qwen run:

```bash
env DOTNET_ROOT=$HOME/.dotnet/zsy-nncase-dotnet8 PATH="$HOME/.dotnet/zsy-nncase-dotnet8:$PATH" LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
  dotnet publish src/Nncase.Compiler/Nncase.Compiler.csproj -c Release --no-restore --sc false -r linux-x64 -o install -v:minimal
cp -f install/lib/*.so install/
```

Default cap Qwen run:

```bash
env NNCASE_CUDA_PE_GMEM_LIMIT_BYTES=589824 TMPDIR=$PWD/tmp/qwen3_cap_589824_final \
  PYTHONPATH=$PWD/tests:$PWD/install/python:$PWD/install:$PWD/install/lib \
  LD_LIBRARY_PATH=$PWD/install:$PWD/install/lib:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
  NNCASE_COMPILER=$PWD/install/Nncase.Compiler.dll \
  DOTNET_ROOT=$HOME/.dotnet/zsy-nncase-dotnet8 \
  PATH="$HOME/.dotnet/zsy-nncase-dotnet8:$PATH" \
  NNCASE_CUDA_FUSED_KERNEL=off \
  NNCASE_CUDA_REQUIRED_PE=16 \
  NNCASE_CUDA_TILE_PE=16 \
  NNCASE_CUDA_USE_NATIVE_TRITON_KERNELS=1 \
  NNCASE_CUDA_REQUIRE_TRITON_KERNELS=1 \
  NNCASE_CUDA_FP32_PARTIALS=1 \
  python -m pytest -vv -s tests/importer/huggingface_/test_qwen3_cuda.py::test_qwen3_cuda_poc
```

## Design Notes

- `NNCASE_CUDA_PE_GMEM_LIMIT_BYTES` is plumbed through CUDA target `MemoryCapacities` and interpreted by AutoDistributed for CUDA only.
- The SAT model adds CUDA op-step memory constraints when the finite cap is present; the cap run above added `8946` such constraints.
- The selected cap `589824` is below the default no-cap picked live tensor peak `8437760` and forces the picked graph down to `589824` bytes.
- The CUDA extraction score ignores memory-load/store as a tie-breaker and emphasizes synchronization for both default no-cap and default cap runs. This is a shared CUDA default cost-model change; the finite cap run additionally adds SAT memory constraints and cap-only candidate penalties.
- Under a finite CUDA cap, large block-local rdata constants and dynamic-sequence-split matmul candidates with fully broadcast static RHS are rejected as generic CUDA gmem-cap candidates. This is target-level behavior, not a Qwen-specific name or layer heuristic.
- No finite CUDA cap path enables const rdata deduplication; metadata/runtime pool changes are secondary diagnostics only.

## Regenerate

```bash
python docs/triton-backend/qwen3-layer0-fused-modes/generate_qwen3_cuda_gmem_shard_comparison.py
```
