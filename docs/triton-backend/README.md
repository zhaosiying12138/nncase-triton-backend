# nncase CUDA/Triton Backend PoC

This document describes the CUDA/Triton backend work in the `nv-triton-codegen`
branch. The current backend is a proof of concept for running an nncase-compiled
NUMA-style graph on an NVIDIA GPU. It is not designed as a high-performance
NVGPU inference backend yet.

## Goals

The main goal is to keep nncase as the graph compiler and use the GPU as an
execution substrate that simulates the Canaan NPU execution model.

The important design points are:

- Keep the nncase graph compiler responsible for graph partitioning, SBP,
  sharding, broadcast, reduce, bufferization, and CCL insertion.
- Treat each CUDA PE as one simulated NPU PE. In the Qwen3 PoC this is normally
  `PE=16`, driven by `NNCASE_CUDA_SM_COUNT=16`.
- Treat GPU memory as PE-local memory pools. The generated runtime code carries
  PE-local data, rdata, thread-local rdata, block-local rdata, and output pools.
- Preserve explicit distributed tensor metadata so the generated runtime knows
  whether an object is sharded, broadcast, partial, local, or collective.
- Simulate CCL as an explicit generated operation, for example
  `gather_reduce_scatter`, instead of silently relying on a global tensor view.
- Prioritize Qwen3-0.6B autoregressive inference over general CNN/Yolo coverage.

Non-goals for this PoC:

- Do not optimize for NVIDIA throughput first.
- Do not replace nncase SBP/AutoDistributed decisions with a GPU-native planner.
- Do not assume a flat GPU memory model in the generated program.
- Do not require AutoVectorize or AutoPacking for the CUDA/Triton target.

## Pass Strategy

The CUDA target is registered as an NTT-shaped target in
`modules/Nncase.Modules.NTT/Targets/CUDATarget.cs`.

Target-independent graph passes remain useful because the CUDA backend still
consumes the distributed graph produced by nncase:

- Shape bucket and dynamic-shape handling.
- AutoDistributed / SBP planning.
- TIR selection.
- Bufferization and memory planning.
- Module partitioning.

CUDA target-specific behavior:

- `EnableAutoVectorize = false`
- `EnableAutoPacking = false`
- The target still uses `NTTAffineSelectionPass` and `NTTTIRSelectionPass`.
- CUDA target options reuse `NTTTargetOptions`, including hierarchy and memory
  architecture fields.

The reason for disabling AutoVectorize/AutoPacking is that Triton is tile-based
and the PoC wants the nncase IR launch granularity to stay close to one
single-PE operator or explicit collective. Extra pack/unpack/vectorization
passes make it harder to preserve that logical execution model and can introduce
layout work that is not part of the NUMA simulation objective.

## NUMA and PE Model

The CUDA target uses NTT hierarchy metadata to decide the simulated PE count.

For Qwen3 CUDA admission:

- `CudaQwenAdmissionRunner` starts from the CUDA SM count or the available
  paged-attention block count.
- It tries PE candidates by halving: for example `16, 8, 4, 2, 1`.
- For each candidate it configures:
  - `Hierarchies = [[PE]]`
  - `HierarchyNames = "p"`
  - `UnifiedMemoryArch = false`
  - `MemoryAccessArch = NUMA`
- The selected PE count is written to
  `tests_output/test_qwen3_cuda_poc/cuda_pe_admission.json`.

The current Qwen3 PoC is expected to run with `PE=16` on the local A2000 setup:

```bash
export NNCASE_CUDA_SM_COUNT=16
```

The PE count is not chosen by measuring final CUDA performance. It is a
simulation parameter for the target architecture.

## Codegen Architecture

The backend codegen path is split into four main layers.

### 1. CUDA Target Registration

Key files:

- `modules/Nncase.Modules.NTT/Targets/CUDATarget.cs`
- `modules/Nncase.Modules.NTT/NTTModule.cs`
- `src/Nncase.Targets/Target.cs`
- `src/Nncase.Core/ITarget.cs`

`CUDATarget` registers the target name `cuda`, provides the module compiler, and
connects the CUDA target to NTT target options and NTT selection passes.

### 2. Launch Collection

Key files:

- `modules/Nncase.Modules.NTT/CodeGen/CUDA/CudaTritonLaunchCollector.cs`
- `modules/Nncase.Modules.NTT/CodeGen/CUDA/CudaTritonSourceModel.cs`

`CudaTritonLaunchCollector` walks each `PrimFunction` and records launches with:

- Launch ordinal.
- Launch kind: `compute`, `memcopy`, `boxing`, `reduce`, `matmul`,
  `collective`, or nested `function`.
- Operation name, for example `matmul`, `tensor_load`, `ro_pe`,
  `gather_reduce_scatter`, `update_paged_attention_kvcache`,
  `paged_attention`.
- Argument names.
- Buffer descriptors: shape, dtype, element size, rank, strides, memory pool,
  span, hierarchy, and distributed type string.
- Return descriptors.

This metadata is used for both generated Python execution and debug logs.

### 3. KModel Embedding

Key files:

- `modules/Nncase.Modules.NTT/CodeGen/CUDA/CudaLinkableModule.cs`
- `modules/Nncase.Modules.NTT/CodeGen/CUDA/CudaSectionNames.cs`

The CUDA linkable module embeds three CUDA-specific sections into the kmodel:

- `.cuda.meta`: JSON metadata for functions, pools, launch descriptors,
  PE count, and buffer descriptors.
- `.triton.module`: marker/module name section.
- `.triton.source`: generated Python source that implements the PoC runtime.

When codegen dump is enabled, the compiler also writes:

- `CodeGen/cuda/triton_module.py`
- `CodeGen/cuda/cuda_meta.json`
- `CodeGen/cuda/launch_summary.txt`

The launch summary is useful for checking static launch structure before
running the simulator.

### 4. Generated Python Runtime

Key file:

- `modules/Nncase.Modules.NTT/CodeGen/CUDA/TritonPythonSourceBuilder.cs`

The generated Python module owns the logical CUDA/Triton execution behavior.
It contains:

- Metadata loader and function registry.
- PE-local runtime context.
- Buffer binding helpers.
- DLPack/PyTorch tensor conversion helpers.
- Multi-PE launch wrappers.
- Fallback implementations for many NTT/TIR operations using PyTorch.
- Simulated collective helpers such as `gather_reduce_scatter`.
- Paged-attention KV update and paged-attention helpers.
- Verbose launch tracing.

In strict Qwen runs, real tensor operators are admitted only if they are handled
by generated Triton native helpers. If an unsupported non-structural operator
would fall back to the older Python/PyTorch path, strict mode raises an error
instead:

```bash
export NNCASE_CUDA_USE_NATIVE_TRITON_KERNELS=1
export NNCASE_CUDA_REQUIRE_TRITON_KERNELS=1
```

Generated modules with `PE>1` default to strict mode even when the environment
variable is not set. Set `NNCASE_CUDA_REQUIRE_TRITON_KERNELS=0` only for
bring-up debugging.

The generated source still contains legacy fallback code for bring-up and
diagnostics, but the Qwen3 CUDA test and the profile runner set strict Triton
mode by default.

The generated module also carries `--fused-kernel` metadata. The accepted modes
are:

- `off`: default baseline. Whole-function fused matching is disabled, but the
  persistent PE-grid Triton kernels remain enabled.
- `compute`: enables CUDA target pass fusion for PE-local compute chains that
  are already backed by native Triton helpers. The fusion pass runs after
  `AutoDistributed` has searched sharding and before CUDA affine
  selection/AutoTiling/TIR lowering, so the fused op does not encode a fixed
  shard strategy. The nncase pass layer now emits
  explicit `Fusion("cuda.*", "cuda", ...)` markers for dense flash attention,
  MLP swish/mul/down-projection chains, `LayerNorm`/RMSNorm-adjacent
  transpose/matmul patterns, RoPE, and RoPE angle `mul -> cos/sin` chains.
  `TIRSelection` lowers those markers to `TIR.NTT.FusedKernel`, and generated
  Python only dispatches the explicit `fusion.cuda.*` op name to the matching
  native helper. Current explicit names include:
  `cuda.flash_attention_*`, `cuda.swish_mul_*`,
  `cuda.matmul_swish_mul_*`, `cuda.silu_mul_matmul_*`,
  `cuda.matmul_mul_matmul_*`,
  `cuda.matmul_silu_matmul_mul_matmul_*`,
  `cuda.layer_norm_transpose_*`, `cuda.layer_norm_matmul_*`,
  `cuda.mul_cos_*`, `cuda.mul_sin_*`, and `cuda.rope_*`. The dense
  flash-attention helper accepts rank3/rank4 tensors with head dimension at
  most 128 and rejects sequence/head-dim cross-PE CCL. The older
  generated-Python whole-function matchers remain only as a transition path for
  uncovered compatibility cases.
- `compute-ccl`: CCL-only tail mode. It keeps the `off` compute launch
  sequence, does not register the CUDA compute-fusion pass, and generated
  metadata should not contain `fusion.cuda.*` compute launches from this mode.
  Codegen instead scans the generated launch metadata for
  `gather_reduce_scatter` data dependencies. A GRS with a unique producer is
  recorded as a producer-side `ccl_tail.<producer>.grs` site and the original
  GRS ordinal is marked `elided_by_ccl_tail=true`; non-unique or unsupported
  sites remain explicit. Function producers are handled by drilling into the
  callee and attaching the tail to the unique nested writer of the parent
  buffer. `tensor_load`, `tensor_store`, `paged_attention`, and KV side-effect
  launches stay explicit.

Dense flash attention and paged attention are separate lowerings. The dense
`QK^T -> scale -> softmax -> AV` compute pattern is fused by the nncase pass
into `Fusion("cuda.flash_attention_*", "cuda", ...)`. Autoregressive Qwen3
uses the paged KV-cache API instead; its `paged_attention` op lowers directly
to `_nncase_paged_flash_attention_rank3_kernel`, a Flash-style online-softmax
Triton kernel adapted to the current Head/Dim/Seq paged KV layout and
owner/slot tables.

There is no separate `tile` fused-kernel mode. Persistent tile execution is the
baseline launch contract and is controlled by `NNCASE_CUDA_TILE_PE`, which
defaults to `16`.

PE-local kernels use a pointer-table launch model. The launch builds one table
entry per PE, and each Triton program owns exactly one logical PE:

```text
grid=(PE,)
pe = tl.program_id(0)
```

Tile work is looped inside that PE program. Rank4 elementwise/copy/transpose,
RoPE, update-kv, CCL materialization, and gather kernels loop over
`MAX_TILES`; matmul-like and dense flash-attention kernels loop over
`MAX_M_TILES`/`MAX_N_TILES`; layer norm loops over `MAX_ROWS`; paged
flash-attention loops over local query-head tiles, query sequence tiles, and
paged KV tiles.

CCL/materialization kernels are explicit Triton launches in `off` and
`compute`. In `compute-ccl`, eligible `gather_reduce_scatter` sites can be
planned as producer-side CCL tails and the original GRS launch is elided. Both
explicit CCL materialization and tail execution receive PE pointer tables and
are the only places that intentionally read data belonging to multiple PEs.
Explicit CCL writes directly to the API-selected destination GMEM buffers.
Tail-planned sites reserve scratch counter bytes through the existing optional
scratch pointer/size ABI.

Verbose logs report two levels. The `[nncase-triton] begin ...` line is the
logical nncase launch. It intentionally reports PE lockstep dispatch, not a
Triton program grid:

```text
logical_pe_dispatch<pe_count=PE, collective=...>
```

The generated module also emits `[nncase-triton-kernel]` lines when verbose
logging is enabled. Those lines report the actual Triton kernel grid and tile
meta, for example `BLOCK=256`, `BLOCK_M/N/K`, or `HEAD_DIM/BLOCK_T/BLOCK_D`.
For persistent helpers the reported grid should be a single-dimensional PE grid,
for example `launch<grid=(16,), ...>` when `PE=16`.

For a layer-by-layer map from Qwen3 model operations to a shard-16 Triton log,
including a Graphviz dataflow graph and ordinal-by-ordinal kernel table, see
[`qwen3-layer0-shard16.md`](qwen3-layer0-shard16.md).

## Native Runtime Architecture

Key files:

- `src/Native/include/nncase/runtime/cuda/runtime_module.h`
- `src/Native/include/nncase/runtime/cuda/runtime_function.h`
- `src/Native/src/runtime/cuda/runtime_module.cpp`
- `src/Native/src/runtime/cuda/runtime_function.cpp`
- `src/Native/src/runtime/cuda/runtime_rdata_layout.cpp`

The native runtime:

- Registers the CUDA runtime module.
- Reads CUDA metadata and generated Python source from the kmodel.
- Initializes Python using `NNCASE_TRITON_PYTHON`.
- Imports the generated Python module.
- Loads PE-local rdata and memory pools.
- Marshals nncase runtime tensors and paged-attention objects into Python.
- Calls the generated `launch(function_id, pe_id, data_pool, output_pool,
  rdata_pool, ...)` entry.

The runtime can cache the generated Python source import. If
`NNCASE_TRITON_CACHE_DIR` is set, the native runtime hashes the embedded
`.triton.source`, writes it as `nncase_triton_<hash>.py`, and imports it through
`importlib`. Python then also produces a normal `__pycache__/*.pyc`.

This cache avoids repeatedly executing a large in-memory Python source string.
It does not change the compiled kmodel itself.

## Verbose Launch Logging

Verbose logging can be enabled by the profile script:

```bash
--verbose
```

or by environment:

```bash
export NNCASE_TRITON_VERBOSE=1
```

Aliases also accepted by the native/generated path:

```bash
export NNCASE_TRITON_VERBOSES=1
export NNCASE_CUDA_VERBOSE=1
```

The verbosity limit is controlled by:

```bash
--verbose-limit 80
export NNCASE_TRITON_VERBOSE_LIMIT=80
```

Example log line:

```text
[nncase-triton] begin function=main_segment_1_prim ordinal=4 kind=collective op=gather_reduce_scatter logical_pe_dispatch<pe_count=16, collective=True> args=[...] attrs={...}
[nncase-triton-kernel] launch function=main_segment_1_prim ordinal=4 kernel=_nncase_ccl_rank4_kernel launch<grid=(16,), block=(256, 1, 1)> pe_count=16 meta={"BLOCK": 256, "MAX_TILES": ..., ...}
```

The native runtime also prints Python-launch timing:

```text
[nncase-cuda] begin python_launch function_id=0 pe_id=0 pe_count=16 arg_count=2 stream=(nil)
[nncase-cuda] end python_launch function_id=0 pe_id=0 elapsed_ms=...
```

## Triton Tile Knobs

The default PoC kernels are tiled, but they are intentionally conservative and
not autotuned. Runtime environment variables can override the tile constants
without rebuilding nncase:

```bash
export NNCASE_TRITON_ELEM_BLOCK=256
export NNCASE_TRITON_CCL_BLOCK=256
export NNCASE_TRITON_MATMUL_BLOCK_M=16
export NNCASE_TRITON_MATMUL_BLOCK_N=32
export NNCASE_TRITON_MATMUL_BLOCK_K=32
```

These values control Triton kernel tile sizes. They are unrelated to the
profile runner's `--block-size`, which controls the paged-attention KV-cache
block size.

## Streaming Token Output

The profile runner supports immediate token streaming:

```bash
--stream-tokens
```

When this option is enabled, every measured decode step prints the newly
generated token immediately after `sim.run()` completes and the token is decoded.
Warmup tokens are not streamed.

The profile runner constructs the paged-attention scheduler to match the kmodel
target. CPU profiling keeps the original `HeadDim` vector lanes used by the
x86 Qwen compile path, while CUDA profiling uses no vector lanes to match
`tests/importer/huggingface_/test_qwen3_cuda.py`.

The final JSON still records the full token list and joined text.

## Compile Cache Modes

The profile runner supports two modes:

```bash
--compile-cache-mode reuse
--compile-cache-mode always
```

`reuse` is the default. It reuses the existing CUDA kmodel:

```text
tests_output/test_qwen3_cuda_poc/infer/cuda/noptq/test.kmodel
```

`always` refreshes CUDA compile artifacts before profiling by running:

```bash
python -m pytest -q -s tests/importer/huggingface_/test_qwen3_cuda.py::test_qwen3_cuda_poc
```

That test path re-runs nncase compile passes, CUDA PE admission, Triton codegen,
and `compiler.gencode`, then leaves the refreshed kmodel under
`tests_output/test_qwen3_cuda_poc/infer/cuda/noptq/test.kmodel`.

This option intentionally recompiles once before the profile run. It does not
recompile before every generated token.

The generated Python source/bytecode import cache is controlled separately:

```bash
--triton-cache-dir tests_output/qwen3_cuda_tokens256/triton_cache
```

or directly:

```bash
export NNCASE_TRITON_CACHE_DIR="$PWD/tests_output/qwen3_cuda_tokens256/triton_cache"
```

## Build and Install

The local validation environment used for this branch is:

```bash
cd /home/zhaosiying/codebase/compiler/nncase
source zsy-nncase/bin/activate

export DOTNET_ROOT="$HOME/.dotnet/zsy-nncase-dotnet8"
export PATH="$PWD/zsy-nncase/bin:$DOTNET_ROOT:$PATH"
export PYTHONPATH="$PWD/install/lib:$PWD/install/python:$PWD/tests"
export LD_LIBRARY_PATH="$PWD/install:$PWD/install/lib:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
export NNCASE_COMPILER="$PWD/install/Nncase.Compiler.dll"
export NNCASE_PLUGIN_PATH="$PWD/install/lib"
export NNCASE_TRITON_PYTHON="$PWD/zsy-nncase/bin/python"
export NNCASE_CUDA_SM_COUNT=16
export NNCASE_CUDA_REQUIRED_PE=16
export NNCASE_CUDA_TILE_PE=16
export NNCASE_CUDA_FUSED_KERNEL=compute-ccl
export NNCASE_CUDA_USE_NATIVE_TRITON_KERNELS=1
export NNCASE_CUDA_REQUIRE_TRITON_KERNELS=1
export NNCASE_CUDA_FP32_PARTIALS=1
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Native/runtime install:

```bash
cmake --build build/Release -j"$(nproc)"
cmake --install build/Release --prefix install
cp -f install/lib/*.so install/
```

Compiler publish:

```bash
dotnet publish src/Nncase.Compiler/Nncase.Compiler.csproj \
  -c Release \
  --no-restore \
  --sc false \
  -r linux-x64 \
  -o install \
  -v:minimal
cp -f install/lib/*.so install/
```

## Compile Qwen3 CUDA KModel

Compile and validate the CUDA Qwen3 PoC kmodel:

```bash
source zsy-nncase/bin/activate

export DOTNET_ROOT="$HOME/.dotnet/zsy-nncase-dotnet8"
export PATH="$PWD/zsy-nncase/bin:$DOTNET_ROOT:$PATH"
export PYTHONPATH="$PWD/install/lib:$PWD/install/python:$PWD/tests"
export LD_LIBRARY_PATH="$PWD/install:$PWD/install/lib:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
export NNCASE_COMPILER="$PWD/install/Nncase.Compiler.dll"
export NNCASE_PLUGIN_PATH="$PWD/install/lib"
export NNCASE_TRITON_PYTHON="$PWD/zsy-nncase/bin/python"
export NNCASE_CUDA_SM_COUNT=16
export NNCASE_CUDA_REQUIRED_PE=16
export NNCASE_CUDA_TILE_PE=16
export NNCASE_CUDA_FUSED_KERNEL=compute-ccl
export NNCASE_CUDA_USE_NATIVE_TRITON_KERNELS=1
export NNCASE_CUDA_REQUIRE_TRITON_KERNELS=1
export NNCASE_CUDA_FP32_PARTIALS=1
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

zsy-nncase/bin/python -m pytest -vv -s \
  tests/importer/huggingface_/test_qwen3_cuda.py::test_qwen3_cuda_poc
```

Expected kmodel:

```text
tests_output/test_qwen3_cuda_poc/infer/cuda/noptq/test.kmodel
```

Expected static codegen artifacts:

```text
tests_output/test_qwen3_cuda_poc/cuda_admission/pe_16/CodeGen/cuda/triton_module.py
tests_output/test_qwen3_cuda_poc/cuda_admission/pe_16/CodeGen/cuda/cuda_meta.json
tests_output/test_qwen3_cuda_poc/cuda_admission/pe_16/CodeGen/cuda/launch_summary.txt
```

For a compute-only fused-kernel regression run, keep strict native Triton mode
enabled and switch only the fused mode:

```bash
export NNCASE_CUDA_FUSED_KERNEL=compute
zsy-nncase/bin/python -m pytest -q -s \
  tests/importer/huggingface_/test_qwen3_cuda.py::test_qwen3_cuda_poc
```

## Run CUDA-Only Profile with Verbose Logs

This command generates 3 tokens, prints detailed CUDA/Triton launch logs, and
streams each token as soon as it is decoded:

```bash
mkdir -p tests_output/qwen3_cuda_verbose3_stream

source zsy-nncase/bin/activate
export DOTNET_ROOT="$HOME/.dotnet/zsy-nncase-dotnet8"
export PATH="$PWD/zsy-nncase/bin:$DOTNET_ROOT:$PATH"
export PYTHONPATH="$PWD/install/lib:$PWD/install/python:$PWD/tests"
export LD_LIBRARY_PATH="$PWD/install:$PWD/install/lib:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
export NNCASE_COMPILER="$PWD/install/Nncase.Compiler.dll"
export NNCASE_PLUGIN_PATH="$PWD/install/lib"
export NNCASE_TRITON_PYTHON="$PWD/zsy-nncase/bin/python"
export NNCASE_CUDA_SM_COUNT=16
export NNCASE_CUDA_REQUIRED_PE=16
export NNCASE_CUDA_TILE_PE=16
export NNCASE_CUDA_FUSED_KERNEL=compute-ccl
export NNCASE_CUDA_USE_NATIVE_TRITON_KERNELS=1
export NNCASE_CUDA_REQUIRE_TRITON_KERNELS=1
export NNCASE_CUDA_FP32_PARTIALS=1
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset NNCASE_TRITON_VERBOSE NNCASE_TRITON_VERBOSES NNCASE_CUDA_VERBOSE

zsy-nncase/bin/python tests/importer/huggingface_/profile_qwen3_runtime.py \
  --targets cuda \
  --tokens 3 \
  --warmup-tokens 0 \
  --fused-kernel compute-ccl \
  --verbose \
  --verbose-limit 40 \
  --stream-tokens \
  --compile-cache-mode reuse \
  --profile-dir tests_output/qwen3_cuda_verbose3_stream \
  --jsonl tests_output/qwen3_cuda_verbose3_stream/metrics.jsonl \
  2>&1 | tee tests_output/qwen3_cuda_verbose3_stream/run.log
```

Observed local small-scale result:

```text
target  pe  prompt_tokens  generated  ttft_s  tpot_s  total_s  tok/s
cuda   16  39            3         10.121  6.683  23.488  0.128

generated text:
<think>
好的
```

## Run CUDA-Only Long Profile

This command generates 256 tokens without verbose launch logs, while still
streaming tokens immediately:

```bash
mkdir -p tests_output/qwen3_cuda_tokens256

source zsy-nncase/bin/activate
export DOTNET_ROOT="$HOME/.dotnet/zsy-nncase-dotnet8"
export PATH="$PWD/zsy-nncase/bin:$DOTNET_ROOT:$PATH"
export PYTHONPATH="$PWD/install/lib:$PWD/install/python:$PWD/tests"
export LD_LIBRARY_PATH="$PWD/install:$PWD/install/lib:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
export NNCASE_COMPILER="$PWD/install/Nncase.Compiler.dll"
export NNCASE_PLUGIN_PATH="$PWD/install/lib"
export NNCASE_TRITON_PYTHON="$PWD/zsy-nncase/bin/python"
export NNCASE_CUDA_SM_COUNT=16
export NNCASE_CUDA_REQUIRED_PE=16
export NNCASE_CUDA_TILE_PE=16
export NNCASE_CUDA_FUSED_KERNEL=compute-ccl
export NNCASE_CUDA_USE_NATIVE_TRITON_KERNELS=1
export NNCASE_CUDA_REQUIRE_TRITON_KERNELS=1
export NNCASE_CUDA_FP32_PARTIALS=1
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset NNCASE_TRITON_VERBOSE NNCASE_TRITON_VERBOSES NNCASE_CUDA_VERBOSE

zsy-nncase/bin/python tests/importer/huggingface_/profile_qwen3_runtime.py \
  --targets cuda \
  --tokens 256 \
  --warmup-tokens 1 \
  --fused-kernel compute-ccl \
  --stream-tokens \
  --compile-cache-mode reuse \
  --profile-dir tests_output/qwen3_cuda_tokens256 \
  --jsonl tests_output/qwen3_cuda_tokens256/metrics.jsonl \
  2>&1 | tee tests_output/qwen3_cuda_tokens256/run.log
```

If 256 tokens OOM on a smaller GPU, reduce `--tokens`, `--num-blocks`, or
`--max-model-len`. The compiled prompt bucket supports prefill sequence length
up to 1024. The profile runner also checks:

```text
prompt_tokens + tokens <= --max-model-len
```

## Recompile Before Profiling

To force nncase passes and Triton codegen before the profile run:

```bash
zsy-nncase/bin/python tests/importer/huggingface_/profile_qwen3_runtime.py \
  --targets cuda \
  --tokens 3 \
  --warmup-tokens 0 \
  --stream-tokens \
  --compile-cache-mode always \
  --profile-dir tests_output/qwen3_cuda_tokens3_recompile
```

Use this when you want to ensure the kmodel is regenerated from the current
source tree. Use `--compile-cache-mode reuse` for normal timing runs.

## Focused Verification

C# CUDA/Triton codegen tests:

```bash
dotnet test src/Nncase.Tests/Nncase.Tests.csproj \
  -c Release \
  --filter "FullyQualifiedName~UnitTestCUDATritonCodeGen" \
  --no-restore
```

Python fused-mode/profile/admission contract tests:

```bash
zsy-nncase/bin/python -m pytest -q \
  tests/other/test_cuda_triton_persistent_contract.py \
  tests/other/test_cuda_qwen_admission.py \
  tests/other/test_qwen3_profile_stream.py
```

End-to-end CUDA Qwen3 PoC:

```bash
zsy-nncase/bin/python -m pytest -vv -s \
  tests/importer/huggingface_/test_qwen3_cuda.py::test_qwen3_cuda_poc
```

## Current Limitations

- The PoC is slower than the CPU backend on the local A2000 because the goal is
  architectural simulation, not NVGPU throughput.
- Qwen3 strict mode is covered by generated Triton helpers, but broad
  CNN/Yolo/general operator coverage is intentionally out of scope.
- The generated runtime still carries dormant Python/PyTorch fallback code for
  diagnostics. Strict Qwen/profile runs set
  `NNCASE_CUDA_REQUIRE_TRITON_KERNELS=1`, so these paths fail fast instead of
  executing real operators.
- Verbose launch logs and token streaming can interleave visually because logs
  and token text both go to stdout/stderr.
- `--compile-cache-mode always` refreshes the Qwen3 CUDA kmodel through the
  existing pytest runner. It is intentionally coarse-grained.
- The implementation is focused on Qwen3-0.6B. CNN/Yolo/general operator
  coverage was not the priority for this branch.
