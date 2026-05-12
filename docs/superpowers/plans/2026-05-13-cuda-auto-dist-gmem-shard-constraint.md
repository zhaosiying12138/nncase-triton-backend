# CUDA AutoDistributed Gmem Shard Constraint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CUDA AutoDistributed per-PE gmem hard constraints change Qwen3-0.6B PE=16 `--fused-kernel=off` prefill layer0 tensor shard strategy, prove the resulting gmem/PE live tensor memory differs, and keep token/accuracy correct.

**Architecture:** Keep the experiment on a fresh branch from `triton-backend/nv-triton-codegen`. The hard constraint must constrain AutoDistributed picked tensor live memory, not CUDA runtime rdata allocation. The final report compares default no-cap vs default cap only, using real Qwen3 metadata and a layer0 graph.

**Tech Stack:** C# AutoDistributed pass and cost model, xUnit tests, Qwen3 pytest CUDA admission demo, Graphviz DOT/SVG docs generation, Python metadata analysis scripts.

---

## Non-Negotiable Requirements

- The final proof must be complete Qwen3-0.6B, PE=16, `NNCASE_CUDA_FUSED_KERNEL=off`.
- The shard strategy difference must be in prefill layer0.
- The final comparison must be default no-cap vs default cap. Experimental cost profiles may be used for debugging but not as the final graph.
- The primary memory metric is AutoDistributed picked-graph per-PE live tensor memory. Runtime CUDA peak and metadata pools are secondary diagnostics only.
- Do not use rdata deduplication as evidence. A finite CUDA memory cap must not implicitly enable constant rdata dedup.
- No Qwen-specific heuristic is allowed. CUDA default cost model changes are allowed if they are target-level and technically justified.
- The old failed worktree/branch must be deleted only after the new branch is pushed successfully.

## File Responsibilities

- `src/Nncase.Passes/Distributed/AutoDistributed.cs`
  - Add CUDA finite memory-cap extraction from `TargetOptions.MemoryCapacities`.
  - Add picked graph live tensor memory accounting.
  - Add hard constraints and solve diagnostics for CUDA per-PE live memory.
  - Apply any CUDA-only cost model changes at the AutoDistributed extraction boundary.

- `src/Nncase.Core/CostModel/Cost.cs`
  - Keep existing test-only `WithMemoryAccessOverride` support.
  - Add only minimal public/internal cost helpers if needed for CUDA cost model changes.

- `src/Nncase.Schedule/Transforms/BufferizePass.cs`
- `src/Nncase.Schedule/Schedule/Bufferize/BufferizeVisitor.cs`
  - Ensure finite CUDA cap does not automatically enable const rdata dedup.
  - If dedup is retained, require an explicit option/env var and keep it off for the main experiment.

- `tests/huggingface_test_runner.py`
  - Add `NNCASE_CUDA_PE_GMEM_LIMIT_BYTES` plumbing into CUDA `NTTTargetOptions.MemoryCapacities`.
  - Record the cap in admission metadata.

- `src/Nncase.Tests/Distributed/UnitTestQwenEmbeddingShardSearch.cs`
  - Extend the existing embedding test to prove the cap prunes high-memory broadcast weight candidates.

- `src/Nncase.Tests/Distributed/UnitTestCudaAutoDistMemoryConstraint.cs`
  - Add matmul/linear minimal proof tests if this keeps the embedding test focused.

- `tests/importer/huggingface_/analyze_qwen3_cuda_memory_cap.py`
  - Create or update a script that compares no-cap/cap metadata for prefill layer0.
  - Compute changed ordinals and per-PE live tensor memory from AutoDistributed dumps.

- `docs/triton-backend/qwen3-layer0-fused-modes/generate_qwen3_cuda_gmem_shard_comparison.py`
  - Generate the final default no-cap vs default cap layer0 DOT/SVG/MD comparison from real metadata.

- `docs/triton-backend/qwen3-cuda-auto-distribute-gmem-shard-constraint.md`
  - Record design, commands, cap search, picked live memory, changed layer0 shard ops, token/accuracy, and limitations.

## Task 1: Clean Worktree Baseline

**Files:**
- Modify: none
- Create: none

- [ ] **Step 1: Verify branch and base**

Run:

```bash
git branch --show-current
git rev-parse HEAD
git rev-parse triton-backend/nv-triton-codegen
git status --short
```

Expected:

- branch is `cuda-auto-dist-gmem-shard-constraint`
- `HEAD` equals latest `triton-backend/nv-triton-codegen`
- worktree is clean except this plan file once added

- [ ] **Step 2: Commit the plan**

Run:

```bash
git add docs/superpowers/plans/2026-05-13-cuda-auto-dist-gmem-shard-constraint.md
git commit -m "docs: plan cuda auto-dist gmem shard constraint"
```

Expected: one docs-only plan commit.

## Task 2: Remove Rdata Dedup Coupling

**Files:**
- Modify: `src/Nncase.Schedule/Transforms/BufferizePass.cs`
- Modify: `src/Nncase.Schedule/Schedule/Bufferize/BufferizeVisitor.cs` only if an explicit dedup switch is needed
- Test: existing build and Qwen metadata checks

- [ ] **Step 1: Inspect current BufferizePass**

Run:

```bash
rg -n "deduplicate|Dedup|MemoryCapacities|ThreadLocalRdata" src/Nncase.Schedule
```

Expected: finite cap must not be coupled to dedup on this fresh base. If coupling is absent, record that no code change is needed.

- [ ] **Step 2: If coupling exists, remove it**

Implementation rule:

```csharp
var bufferizeVisitor = new BufferizeVisitor(funcs);
```

Do not pass `ShouldDeduplicateConstRdata(funcs.Key)` from finite `MemoryCapacities`.

- [ ] **Step 3: Add explicit opt-in only if current code needs dedup retained**

If retained, use a new default-off env var:

```text
NNCASE_CUDA_DEDUP_CONST_RDATA=1
```

The main Qwen experiment must run with this unset.

- [ ] **Step 4: Verify no finite-cap dedup path remains**

Run:

```bash
rg -n "MemoryCapacities.*dedup|ShouldDeduplicateConstRdata|NNCASE_CUDA_DEDUP_CONST_RDATA" src/Nncase.Schedule
```

Expected: no automatic finite-cap dedup; any dedup path is explicit opt-in.

## Task 3: Implement CUDA Per-PE Cap Plumbing

**Files:**
- Modify: `tests/huggingface_test_runner.py`
- Modify: `tests/other/test_cuda_qwen_admission.py`
- Modify: `src/Nncase.Passes/Distributed/AutoDistributed.cs`
- Test: `tests/other/test_cuda_qwen_admission.py`

- [ ] **Step 1: Add `NNCASE_CUDA_PE_GMEM_LIMIT_BYTES` to Python target options**

In `CudaQwenAdmissionRunner.make_cuda_pe_target_options`, after fused-kernel mode:

```python
pe_gmem_limit = os.getenv("NNCASE_CUDA_PE_GMEM_LIMIT_BYTES", "").strip()
if pe_gmem_limit:
    try:
        pe_gmem_limit_value = int(pe_gmem_limit)
    except ValueError as ex:
        raise CudaAdmissionUnavailable(
            f"NNCASE_CUDA_PE_GMEM_LIMIT_BYTES must be an integer, got {pe_gmem_limit!r}") from ex
    if pe_gmem_limit_value <= 0:
        raise CudaAdmissionUnavailable(
            f"NNCASE_CUDA_PE_GMEM_LIMIT_BYTES must be positive, got {pe_gmem_limit}")
    options.MemoryCapacities = [524288, pe_gmem_limit_value]
```

- [ ] **Step 2: Record the cap in admission record**

In `_compile_cuda_candidate`, add `MemoryCapacities` and `CudaPeGmemLimitBytes` to `target_options` only when the env var is set.

- [ ] **Step 3: Add Python admission tests**

In `tests/other/test_cuda_qwen_admission.py`, add tests for:

```python
monkeypatch.setenv("NNCASE_CUDA_PE_GMEM_LIMIT_BYTES", str(64 * 1024 * 1024))
options = runner.make_cuda_pe_target_options(16)
assert list(options.MemoryCapacities) == [524288, 64 * 1024 * 1024]
```

and invalid `0` raises `CudaAdmissionUnavailable`.

- [ ] **Step 4: Add CUDA cap extraction in AutoDistributed**

Add helper:

```csharp
private static bool TryGetSingleNodeMemoryLimitBytes(string moduleKind, INTTTargetOptions targetOptions, out long limitBytes)
{
    limitBytes = 0;
    if (moduleKind == CUDATarget.Kind && targetOptions.MemoryCapacities.Length > 0)
    {
        var capacity = targetOptions.MemoryCapacities[^1];
        if (capacity > 0 && capacity < int.MaxValue)
        {
            limitBytes = capacity;
            return true;
        }
    }

    // Preserve existing XPU path unchanged.
    ...
}
```

- [ ] **Step 5: Apply cap to leaf and boxing candidate generation**

Use `SingleNodeMemoryCheck` in:

- `GetLeafCandidateDistTypes`
- `BuildEquivalentCalls` reshape branch
- `GetDiverseCandidateSBPs`

Expected: over-limit candidate distributed types are not inserted when cap is finite.

## Task 4: Add Picked-Graph Live Tensor Memory Metrics

**Files:**
- Modify: `src/Nncase.Passes/Distributed/AutoDistributed.cs`
- Test: `src/Nncase.Tests/Distributed/UnitTestQwenEmbeddingShardSearch.cs`

- [ ] **Step 1: Add local-size helpers**

Implement:

```csharp
private static long GetDistributedLocalSize(IRType type)
```

for `DistributedType` and `TupleType`, using `DistributedUtility.GetDividedTensorType`.

- [ ] **Step 2: Add op-local footprint helper**

Implement:

```csharp
private long GetNodeLocalFootprintBytes(SearchableNode node)
```

This should include the node output local size plus local sizes of direct data-dependency inputs available through `_rootSearchGraph` edges. Do not count constants as an aggregate model-size cap.

- [ ] **Step 3: Add picked max diagnostics**

After solve, compute:

```csharp
SingleNodeMemoryPickedMaxBytes
ConstMemoryPickedBytes
```

`ConstMemoryPickedBytes` is diagnostic only and must not constrain the model.

- [ ] **Step 4: Add segment live peak dump**

For the picked graph, produce a deterministic diagnostic in `Costs/Solve.txt`:

```text
PickedLocalTensorLivePeakBytes : <value>
PickedLocalTensorLivePeakNode : <node-id/op/type summary>
PickedShardSignatureHash : <hash>
```

If exact liveness is too complex for the first commit, compute a conservative per-node live upper bound and name it `PickedLocalTensorNodePeakBytes`; add exact segment liveness in the next task before Qwen validation.

- [ ] **Step 5: Verify solve dump includes metrics**

Run focused xUnit test with dump enabled if available, or inspect a generated `Solve.txt`.

Expected: no-cap and cap runs show memory enabled/limit/picked metrics.

## Task 5: Minimal Proof Tests

**Files:**
- Modify: `src/Nncase.Tests/Distributed/UnitTestQwenEmbeddingShardSearch.cs`
- Create or modify: `src/Nncase.Tests/Distributed/UnitTestCudaAutoDistMemoryConstraint.cs`

- [ ] **Step 1: Extend embedding test helper to accept cap**

Change:

```csharp
private async Task<(Function Result, DimVar SequenceLength)> RunAutoDistributedAsync(long? cudaPeMemoryLimitBytes = null)
```

When cap is set:

```csharp
targetOptions.MemoryCapacities = [524288, checked((int)cudaPeMemoryLimitBytes.Value)];
```

- [ ] **Step 2: Add embedding cap test**

Use the existing test-only memory access override to make broadcast weight attractive without cap. Then set a cap below broadcast weight local size and assert the gather weight is sharded:

```csharp
using var memoryAccessOverride = CostUtility.WithMemoryAccessOverride(raw => raw == 0 ? 0 : 1024);
var (result, sequenceLength) = await RunAutoDistributedAsync(cudaPeMemoryLimitBytes: 64L * 1024L * 1024L);
Assert.Equal(HiddenShardWeightType(placement), Assert.IsType<DistributedType>(gather.Arguments[0].CheckedType));
```

- [ ] **Step 3: Add matmul/linear cap test**

Build a small linear graph with large `[hidden, intermediate]` weight and an output scheme that allows both replicated and sharded weight plans. Assert:

- no cap picks a high-memory/low-communication strategy under CUDA default cost model or test override;
- finite cap picks a lower-memory strategy;
- output distributed type still matches requested scheme.

- [ ] **Step 4: Run focused tests**

Run:

```bash
export DOTNET_ROOT=$HOME/.dotnet/zsy-nncase-dotnet8
export PATH=$DOTNET_ROOT:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu
dotnet test src/Nncase.Tests/Nncase.Tests.csproj -c Release --no-restore --filter "FullyQualifiedName~UnitTestQwenEmbeddingShardSearch|FullyQualifiedName~UnitTestCudaAutoDistMemoryConstraint" --logger "console;verbosity=normal"
```

Expected: all focused tests pass.

## Task 6: CUDA Default Cost Model Revision

**Files:**
- Modify: `src/Nncase.Passes/Distributed/AutoDistributed.cs`
- Modify cost evaluators only if a target-independent bug is found
- Test: focused unit tests and Qwen compile diagnostics

- [ ] **Step 1: Dump default Qwen no-cap picked cost and shard signatures**

Run Qwen no-cap with dump enabled and collect:

- `Costs/Solve.txt`
- picked shard signature for `main_segment_1_prim` ord `0..34`
- `PickedLocalTensorLivePeakBytes`

- [ ] **Step 2: Identify why default no-cap does not pick high-memory strategy**

Inspect `AutoDistributedSearch.md` and candidate costs near layer0 ops. Determine whether the current default cost model over-penalizes memory or under-penalizes CCL/boxing in CUDA.

- [ ] **Step 3: Implement CUDA-only cost model revision**

Acceptable changes:

- increase CUDA communication/boxing synchronization cost if it is clearly underweighted;
- scale CUDA memory traffic cost by bytes consistently where existing code counts elements inconsistently;
- add CUDA-only extraction cost weights inside AutoDistributed.

Unacceptable changes:

- no Qwen op-name checks;
- no layer index checks;
- no hard-coded Qwen dimensions.

- [ ] **Step 4: Preserve non-CUDA behavior**

Guard changes with:

```csharp
if (_moduleKind == CUDATarget.Kind) { ... }
```

or use target options so XPU/CPU paths remain unchanged.

- [ ] **Step 5: Re-run focused tests**

Same command as Task 5. Expected: tests pass and minimal tests still show cap-induced strategy change.

## Task 7: Qwen3 Default No-Cap vs Default Cap Search

**Files:**
- Generate under: `tests_output/qwen3_off_default_nocap_*`
- Generate under: `tests_output/qwen3_off_default_cap_<bytes>_*`
- Modify scripts only if needed

- [ ] **Step 1: Build and install**

Run:

```bash
export DOTNET_ROOT=$HOME/.dotnet/zsy-nncase-dotnet8
export PATH=$DOTNET_ROOT:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu
dotnet publish src/Nncase.Compiler/Nncase.Compiler.csproj -c Release --no-restore --sc false -r linux-x64 -o install -v:minimal
cp -f install/lib/*.so install/
```

Expected: publish succeeds.

- [ ] **Step 2: Run Qwen no-cap baseline**

Run with:

```bash
unset NNCASE_CUDA_PE_GMEM_LIMIT_BYTES
export NNCASE_CUDA_FUSED_KERNEL=off
export NNCASE_CUDA_REQUIRED_PE=16
export NNCASE_CUDA_TILE_PE=16
export NNCASE_CUDA_USE_NATIVE_TRITON_KERNELS=1
export NNCASE_CUDA_REQUIRE_TRITON_KERNELS=1
export NNCASE_CUDA_FP32_PARTIALS=1
```

Use the existing Qwen pytest command:

```bash
python -m pytest -vv -s tests/importer/huggingface_/test_qwen3_cuda.py::test_qwen3_cuda_poc
```

Expected: token ratio `1.0000`.

- [ ] **Step 3: Compute no-cap layer0 live peak**

Parse `Solve.txt` and metadata. Record:

- `PickedLocalTensorLivePeakBytes`
- changed candidate signatures available for cap search
- layer0 ord `0..34` shard signature

- [ ] **Step 4: Search cap**

Start from slightly below no-cap layer0 live peak. Use bisection or targeted scan:

```bash
export NNCASE_CUDA_PE_GMEM_LIMIT_BYTES=<candidate>
python -m pytest -vv -s tests/importer/huggingface_/test_qwen3_cuda.py::test_qwen3_cuda_poc
```

Stop only when all are true:

- compile succeeds;
- token ratio `1.0000`;
- prefill layer0 shard signature differs from no-cap;
- cap run picked live peak is lower than no-cap by AutoDistributed live tensor metric.

- [ ] **Step 5: If no cap changes layer0, revise Task 6**

Do not accept a negative result. Because final acceptance requires full Qwen layer0 change, return to cost model revision or hard-constraint design.

## Task 8: Metadata Analyzer and Graph Generation

**Files:**
- Create: `tests/importer/huggingface_/analyze_qwen3_cuda_memory_cap.py`
- Create: `docs/triton-backend/qwen3-layer0-fused-modes/generate_qwen3_cuda_gmem_shard_comparison.py`
- Create: `docs/triton-backend/qwen3-layer0-fused-modes/qwen3-off-vs-off-gmem-cap-layer0.{dot,svg,md}`
- Modify: `docs/triton-backend/qwen3-layer0-fused-modes/README.md`

- [ ] **Step 1: Analyzer inputs**

The analyzer accepts:

```bash
--left tests_output/<nocap>/cuda_admission/pe_16
--right tests_output/<cap>/cuda_admission/pe_16
--function main_segment_1_prim
--ordinals 0:35
```

- [ ] **Step 2: Analyzer output**

It prints JSON with:

```json
{
  "changed_ordinals": [4, 6],
  "left_live_peak_bytes": 123,
  "right_live_peak_bytes": 456,
  "left_token_ratio": "1.0000",
  "right_token_ratio": "1.0000"
}
```

- [ ] **Step 3: Graph requirements**

The DOT/SVG must show:

- left default no-cap;
- right default cap;
- dashed ordinal mappings;
- red boxes around changed ordinals;
- per-op `in_type/out_type` SBP for changed ops;
- live tensor peak summary in graph legend.

- [ ] **Step 4: Generate docs**

Run:

```bash
python docs/triton-backend/qwen3-layer0-fused-modes/generate_qwen3_cuda_gmem_shard_comparison.py
dot -Tsvg docs/triton-backend/qwen3-layer0-fused-modes/qwen3-off-vs-off-gmem-cap-layer0.dot -o docs/triton-backend/qwen3-layer0-fused-modes/qwen3-off-vs-off-gmem-cap-layer0.svg
```

Expected: graph has at least one changed prefill layer0 ordinal.

## Task 9: Final Report

**Files:**
- Create: `docs/triton-backend/qwen3-cuda-auto-distribute-gmem-shard-constraint.md`
- Modify: `docs/triton-backend/qwen3-layer0-fused-modes/README.md`

- [ ] **Step 1: Report commands**

Include exact commands for:

- build/publish;
- no-cap Qwen run;
- cap Qwen run;
- analyzer;
- graph generation;
- focused tests.

- [ ] **Step 2: Report proof table**

Include:

| Metric | no cap | cap | delta |
| --- | ---: | ---: | ---: |
| layer0 changed ordinals | list | list | n/a |
| picked live tensor peak bytes | value | value | reduction |
| token ratio | 1.0000 | 1.0000 | n/a |

- [ ] **Step 3: Report excluded evidence**

State explicitly:

- rdata dedup is off/unset for the main proof;
- runtime peak is secondary and not the proof;
- metadata rdata pools are not used as the main memory metric.

## Task 10: Verification, Commits, Push, Cleanup

**Files:**
- All touched files

- [ ] **Step 1: Run final focused tests**

Run:

```bash
dotnet test src/Nncase.Tests/Nncase.Tests.csproj -c Release --no-restore --filter "FullyQualifiedName~UnitTestQwenEmbeddingShardSearch|FullyQualifiedName~UnitTestCudaAutoDistMemoryConstraint" --logger "console;verbosity=normal"
```

Expected: all pass.

- [ ] **Step 2: Run final Qwen no-cap and cap verification**

Expected:

- both pass token ratio `1.0000`;
- cap run has changed prefill layer0 shard signature;
- cap run has lower picked live tensor peak.

- [ ] **Step 3: Run docs checks**

Run:

```bash
python -m py_compile tests/importer/huggingface_/analyze_qwen3_cuda_memory_cap.py docs/triton-backend/qwen3-layer0-fused-modes/generate_qwen3_cuda_gmem_shard_comparison.py
git diff --check
```

Expected: no errors.

- [ ] **Step 4: Commit logical slices**

Use separate commits:

```bash
git commit -m "cuda: add auto-dist live gmem constraint"
git commit -m "tests: prove cuda auto-dist gmem shard switching"
git commit -m "cuda: tune auto-dist cost for gmem-constrained qwen"
git commit -m "docs: add qwen3 gmem shard constraint evidence"
```

Use only commits that match actual changed file sets.

- [ ] **Step 5: Push new branch**

Run:

```bash
git push -u triton-backend cuda-auto-dist-gmem-shard-constraint
```

- [ ] **Step 6: Delete old failed branch/worktree**

Only after push succeeds:

```bash
git worktree remove /home/zhaosiying/.config/superpowers/worktrees/nncase-auto-dist-cuda-memory-constraint/nncase
git branch -D auto-dist-cuda-memory-constraint
git push triton-backend --delete auto-dist-cuda-memory-constraint
```

Expected: old failed local worktree, local branch, and remote branch are removed.
