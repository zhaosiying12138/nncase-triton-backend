# Qwen3 embedding shard 搜索：从 metadata 读出的一次 all-to-all

这篇笔记只记录当前 Qwen3 CUDA/Triton PoC 的一个很窄证据点：embedding 之后，AutoDistributed 选出的 SBP 并不是把 embedding 当作一块 flat GPU tensor 直接读完，而是在 `main_segment_1_prim` 开头显式留下了一次 reshard。

当前证据来自本地工作区两个已生成的 `cuda_meta.json`。`tests_output/` 是测试生成目录，不是 clean checkout 自带文件；本文下面嵌入了关键摘录，路径用于在当前工作区复查或重新生成后比对。

- `tests_output/qwen3_reshard_off_probe2/cuda_admission/pe_16/CodeGen/cuda/cuda_meta.json`
- `tests_output/qwen3_reshard_compute_ccl_probe/cuda_admission/pe_16/CodeGen/cuda/cuda_meta.json`

`cuda_meta.json` 的 launch/buffer 描述来自 CUDA codegen 侧的
`modules/Nncase.Modules.NTT/CodeGen/CUDA/CudaTritonLaunchCollector.cs`；AutoDistributed pass 的调度入口在
`src/Nncase.Compiler/Compiler.cs` 的 `AutoDistributedPass`。

这两个 probe 的 fused mode 不同，但 `main_segment_1_prim` 的 embedding 入口形态一致：先把 `input_ids` 装载成 broadcast，随后 gather 出 hidden shard，最后把 `(B,S(0))` 变成 `(S(0),B)`。

## 复现 metadata 摘录

在当前工作区可以直接用这些 dump 复查，不需要重跑编译；如果换成干净 checkout，需要先按后面的命令重新生成 `tests_output/`：

```bash
jq '.functions[]
  | select(.name == "main_segment_1_prim")
  | .launches[]
  | select(.ordinal >= 2 and .ordinal <= 4)
  | {
      ordinal,
      kind,
      op_name,
      arguments,
      op_attrs,
      types: (
        .buffer_arguments
        | to_entries
        | map({key, name: .value.name, distributed_type: .value.distributed_type})
        | map(select(.distributed_type != null))
      )
    }' \
  tests_output/qwen3_reshard_off_probe2/cuda_admission/pe_16/CodeGen/cuda/cuda_meta.json
```

把最后一个路径换成
`tests_output/qwen3_reshard_compute_ccl_probe/cuda_admission/pe_16/CodeGen/cuda/cuda_meta.json`
可以得到同一段结构。

如果需要重新生成 Qwen3 CUDA admission dump，当前测试入口是：

```bash
NNCASE_CUDA_SM_COUNT=16 \
NNCASE_CUDA_REQUIRED_PE=16 \
NNCASE_CUDA_USE_NATIVE_TRITON_KERNELS=1 \
NNCASE_CUDA_REQUIRE_TRITON_KERNELS=1 \
NNCASE_CUDA_FP32_PARTIALS=1 \
NNCASE_CUDA_FUSED_KERNEL=compute-ccl \
python -m pytest -vv -s tests/importer/huggingface_/test_qwen3_cuda.py
```

## main_segment_1_prim 的关键三步

两份 metadata 对 buffer 编号不完全一样，但语义相同：

1. `ord02` 是 `tensor_load`，把 `input_ids` 装载成分布式输入：`i64[S] SBP=(B)`。
2. `ord03` 是 `gather(axis=0)`，embedding 表是 `f16[151936,1024] SBP=(B,S(0))`，索引是 `i64[S] SBP=(B)`，输出是 `f16[S,1024] SBP=(B,S(0))`。
3. `ord04` 是 collective `gather_reduce_scatter`。这里把它按语义命名为 all-to-all reshard：`f16[S,1024] SBP=(B,S(0)) -> f16[S,1024] SBP=(S(0),B)`。

这一步是整个片段最值得保留下来的证据：embedding gather 之后，hidden 维仍是 PE 间 split，而后续层希望沿 sequence 维切分、hidden 维 broadcast，于是 metadata 明确出现一次 collective reshard。`gather_reduce_scatter` 是当前 runtime/codegen 的实现名；博客里说 all-to-all，是为了突出它在 SBP 视角下完成的是跨 PE 重新排布。

## 为什么这是搜索，不是手写规则

AutoDistributed 的思路可以类比 e-graph / egg-saturate，但当前实现不是直接调用通用 `EGraphExtractor`。

通用 e-graph 侧的代码在 `src/Nncase.EGraph/Passes/EGraphExtractor.cs`：它为 `ENode` 建 bool 变量、加根节点和子节点约束，用 OR-Tools CP-SAT 在 cost objective 下抽取表达式。AutoDistributed 借用了相同的“枚举候选再用 SAT 抽取最低代价方案”的思想，但具体数据结构是自己的 `DistributedSearchGraph`，核心在 `src/Nncase.Passes/Distributed/AutoDistributed.cs`。

可以按五层理解：

1. 候选枚举：leaf tensor、const、op output 会枚举可行的 distributed type / SBP；call 侧会对输入 bucket 做 Cartesian product，构造等价 call，再靠 type inference 过滤不合法组合。
2. bucket / cluster：`DistributedSearchGraph` 里，cluster 表示一个 IR value 或 call 的候选集合；bucket 通常按候选 checked type / SBP 分组。边表示“如果选这个 op 候选，就必须从某个 input bucket 里选输入候选”。
3. `ExactlyOne` 约束：`SolveAndExtract` 为每个 `SearchableNode` 建 bool var；root 必须只选一个候选，每个 cluster 也只选一个真实 op 候选；当某个候选被选中时，每个 input index 也必须选一个 child 候选。
4. cost objective：普通节点给极小基础 cost，op 节点用 `DistributedCostEvaluateContext` 走 evaluator，boxing / collective 会把通信、memory load/store 等因素放进 score。最终目标是 `WeightedSum(boolVar, costScore)` 最小。
5. OR-Tools CP-SAT extract：求解后得到一组 picked nodes，再由 `ExprBuildVisitor` 把选中的 SBP、boxing 和 op candidate 重建成最终 IR。

所以可以说它“像 egg-saturate”：先把许多等价 shard 方案放进图，再做约束抽取；但要明确，它实际跑的是 AutoDistributed search graph，不是把 Qwen graph 交给通用 e-graph extractor。

## unittest solver log 摘录

新增的最小复现测试在
`src/Nncase.Tests/Distributed/UnitTestQwenEmbeddingShardSearch.cs`。IR 只保留 embedding gather 本身，再接一个 `Unary(Abs)` 作为“后续 op 需要 `SBP=(S(0),B)`”的锚点。这个锚点不改变要观察的核心路径：solver 仍然需要在 gather 后决定是否插入 `SBP=(B,S(0)) -> SBP=(S(0),B)` 的 reshard。

### 测试方法与预期结果

最小复现只需要跑 focused xUnit：

```bash
dotnet test src/Nncase.Tests/Nncase.Tests.csproj \
  -c Debug \
  --filter FullyQualifiedName~UnitTestQwenEmbeddingShardSearch
```

如果本机 `dotnet` 不在 `PATH`，或者 conda 的 `LD_LIBRARY_PATH` 影响 OR-Tools native library，可以用我本地验证过的等价命令：

```bash
env -u LD_LIBRARY_PATH ~/.dotnet/zsy-nncase-dotnet8/dotnet test \
  src/Nncase.Tests/Nncase.Tests.csproj \
  -c Debug \
  --filter FullyQualifiedName~UnitTestQwenEmbeddingShardSearch \
  --no-restore
```

这个测试类有两个用例：

1. `TestBaselineSearchKeepsEmbeddingHiddenShardThenReshardsToSequenceShard`
   - 预期 `input_ids` 是 `i64[sequence_length] SBP=(B)`。
   - 预期 embedding weight 是 `f16[151936,1024] SBP=(B,S(0))`。
   - 预期 `Gather(axis=0)` 输出 `f16[sequence_length,1024] SBP=(B,S(0))`。
   - 预期存在 `Boxing`：`SBP=(B,S(0)) -> SBP=(S(0),B)`，对应 metadata 中的 `ord04 all-to-all reshard`。
2. `TestFlatMemoryCostAvoidsHiddenShardToSequenceShardBoxing`
   - 测试内把所有非零 memory access 归一到同一代价。
   - 预期 `Gather(axis=0)` 改选 `f16[sequence_length,1024] SBP=(B,B)`。
   - 预期存在替代 `Boxing`：`SBP=(B,B) -> SBP=(S(0),B)`。
   - 预期不再存在 `SBP=(B,S(0)) -> SBP=(S(0),B)` 的 hidden-shard all-to-all。

baseline dump 在
`tests_output/UnitTestQwenEmbeddingShardSearch/TestBaselineSearchKeepsEmbeddingHiddenShardThenReshardsToSequenceShard/0_AutoDistributedPass/main/Costs/`。

`Solve.txt` 的关键行：

```text
Status : Optimal
ModelScale :
  Clusters : 7
  Buckets : 14
  CandidateNodes : 15
  SearchEdges : 16
ConstraintSummary :
  RootExactlyOne : 1
  ClusterExactlyOne : 2
  ChildInputExactlyOne : 16
FinalPickedTotalCost : CPUCycles=4099, MemoryLoad=19456128, MemoryStore=9743936, Synchronization=50000, Score=29254163
```

`AutoDistributedSearch.md` 里，solver 选中的是：

```text
N9 picked: Gather(axis=0)
  IR type: f16[sequence_length,1024], (B,S(0)), [p:16]
  input weight bucket: f16[151936,1024], (B,S(0)), [p:16]

N11 picked: Boxing
  NewType: f16[sequence_length,1024], (S(0),B), [p:16]
  input bucket: f16[sequence_length,1024], (B,S(0)), [p:16]
  Cost: CPUCycles=1, MemoryStore=7680, Synchronization=25000, Score=32681
```

这正是 metadata 中 `ord03 gather` 后接 `ord04 all-to-all reshard` 的 IR 级复现。

## flat memory cost 对照结果

第二个测试用 `CostUtility.WithMemoryAccessOverride(raw => raw == 0 ? 0 : 1024)` 把所有非零 memory access 归一到同一个 1024 元素粒度。也就是说，加载 64 个元素和加载 1024 个元素在这个实验里同价。

这次 dump 在
`tests_output/UnitTestQwenEmbeddingShardSearch/TestFlatMemoryCostAvoidsHiddenShardToSequenceShardBoxing/0_AutoDistributedPass/main/Costs/`。

`Solve.txt` 仍然是最优解，但目标函数已经变了：

```text
Status : Optimal
FinalPickedTotalCost : CPUCycles=4100, MemoryLoad=6144, MemoryStore=5120, Synchronization=50000, Score=65364
```

更关键的是 picked path 变了：

```text
N8 picked: Gather(axis=0)
  IR type: f16[sequence_length,1024], (B,B), [p:16]
  input weight bucket: f16[151936,1024], (B,B), [p:16]

N10 picked: Boxing
  NewType: f16[sequence_length,1024], (S(0),B), [p:16]
  input bucket: f16[sequence_length,1024], (B,B), [p:16]

N11 not picked:
  input bucket would have been f16[sequence_length,1024], (B,S(0)), [p:16]
  Cost: CPUCycles=1, MemoryStore=1920, Synchronization=25000, Score=26921
```

对比很直接：

- 当前真实 cost：`ord03 gather` 后保持 hidden shard，`SBP=(B,S(0))`，再用 `ord04` all-to-all reshard 到 `SBP=(S(0),B)`。
- flat memory cost：embedding weight 不再按 hidden 维切分，gather 选择 `SBP=(B,B)`，随后只做 `SBP=(B,B) -> SBP=(S(0),B)`，没有选择 hidden-shard-to-sequence-shard all-to-all。
- 含义：这个例子不是手写规则，而是同一组候选和约束在不同 cost 参数下自动抽取出不同方案。只要系统认为“少读 hidden 列”有价值，hidden shard + all-to-all 就合理；一旦 64 和 1024 的 load 被视为同价，这个优势消失，solver 就会换解。

顺带修正了 `BoxingEvaluator` 里 reshard cost 的一个细节：split 轴变化的 gather/scatter part 应按真实 hierarchy 大小累计。对 `PE=16` 来说，`S(0)` 不是 1 份，而是 16 份；否则 all-to-all 的 memory cost 会被低估。
