// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Distributed;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.DistributedTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestQwenEmbeddingShardSearch : TestClassBase
{
    private const string OutputName = "embed_out";
    private const string HierarchyName = "p";

    public UnitTestQwenEmbeddingShardSearch()
    {
        DefaultTargetName = CUDATarget.Kind;
#if DEBUG
        CompileOptions.DumpFlags = DumpFlags.PassIR | DumpFlags.Rewrite | DumpFlags.EGraphCost | DumpFlags.Compile;
#endif
    }

    [Fact]
    public async Task TestBaselineSearchKeepsEmbeddingHiddenShardThenReshardsToSequenceShard()
    {
        var (result, sequenceLength) = await RunAutoDistributedAsync();
        var placement = Placement();
        var hiddenShard = HiddenShardOutputType(sequenceLength, placement);
        var sequenceShard = SequenceShardOutputType(sequenceLength, placement);

        var gather = Assert.Single(FindCalls(result.Body, call => call.Target is IR.Tensors.Gather));
        Assert.Equal(HiddenShardWeightType(placement), Assert.IsType<DistributedType>(gather.Arguments[0].CheckedType));
        Assert.Equal(BroadcastInputIdsType(sequenceLength, placement), Assert.IsType<DistributedType>(gather.Arguments[1].CheckedType));
        Assert.Equal(hiddenShard, Assert.IsType<DistributedType>(gather.CheckedType));
        AssertHasBoxing(result.Body, hiddenShard, sequenceShard);
    }

    [Fact]
    public async Task TestCudaExtractionScoreDoesNotAdmitBroadcastEmbeddingFromMemoryOverride()
    {
        using var memoryAccessOverride = CostUtility.WithMemoryAccessOverride(raw => raw == 0 ? (UInt128)0 : (UInt128)1024);

        var (result, sequenceLength) = await RunAutoDistributedAsync();
        var placement = Placement();
        var hiddenShard = HiddenShardOutputType(sequenceLength, placement);
        var sequenceShard = SequenceShardOutputType(sequenceLength, placement);

        var gather = Assert.Single(FindCalls(result.Body, call => call.Target is IR.Tensors.Gather));
        Assert.Equal(HiddenShardWeightType(placement), Assert.IsType<DistributedType>(gather.Arguments[0].CheckedType));
        Assert.Equal(BroadcastInputIdsType(sequenceLength, placement), Assert.IsType<DistributedType>(gather.Arguments[1].CheckedType));
        Assert.Equal(hiddenShard, Assert.IsType<DistributedType>(gather.CheckedType));
        AssertHasBoxing(result.Body, hiddenShard, sequenceShard);

        AssertDoesNotHaveBoxing(
            result.Body,
            BroadcastOutputType(sequenceLength, placement),
            sequenceShard);
    }

    [Fact]
    public async Task TestCudaPerPeMemoryConstraintPrunesBroadcastEmbeddingWeight()
    {
        using var memoryAccessOverride = CostUtility.WithMemoryAccessOverride(raw => raw == 0 ? (UInt128)0 : (UInt128)1024);

        var (result, sequenceLength) = await RunAutoDistributedAsync(64L * 1024L * 1024L);
        var placement = Placement();
        var hiddenShard = HiddenShardOutputType(sequenceLength, placement);
        var sequenceShard = SequenceShardOutputType(sequenceLength, placement);

        var gather = Assert.Single(FindCalls(result.Body, call => call.Target is IR.Tensors.Gather));
        Assert.Equal(HiddenShardWeightType(placement), Assert.IsType<DistributedType>(gather.Arguments[0].CheckedType));
        Assert.Equal(BroadcastInputIdsType(sequenceLength, placement), Assert.IsType<DistributedType>(gather.Arguments[1].CheckedType));
        Assert.Equal(hiddenShard, Assert.IsType<DistributedType>(gather.CheckedType));
        AssertHasBoxing(result.Body, hiddenShard, sequenceShard);
        AssertDoesNotHaveBoxing(result.Body, BroadcastOutputType(sequenceLength, placement), sequenceShard);
    }

    [Fact]
    public async Task TestCudaMemoryCapPrunesBroadcastOutputWithStaticConstWeight()
    {
        var function = CreateConstEmbeddingFunction(out var sequenceLength);
        var result = await RunAutoDistributedAsync(function, "const_embed_out", 128L * 1024L);
        var placement = Placement();
        var hiddenShard = SmallHiddenShardOutputType(sequenceLength, placement);
        var sequenceShard = SmallSequenceShardOutputType(sequenceLength, placement);

        var gather = Assert.Single(FindCalls(result.Body, call => call.Target is IR.Tensors.Gather));
        Assert.Equal(SmallHiddenShardWeightType(placement), Assert.IsType<DistributedType>(gather.Arguments[0].CheckedType));
        Assert.Equal(SmallBroadcastInputIdsType(sequenceLength, placement), Assert.IsType<DistributedType>(gather.Arguments[1].CheckedType));
        Assert.Equal(hiddenShard, Assert.IsType<DistributedType>(gather.CheckedType));
        AssertHasBoxing(result.Body, hiddenShard, sequenceShard);
        AssertDoesNotHaveBoxing(result.Body, SmallBroadcastOutputType(sequenceLength, placement), sequenceShard);
    }

    [Fact]
    public async Task TestCudaMemoryCapRejectsLargeBlockLocalRdataConstCandidate()
    {
        using var memoryAccessOverride = CostUtility.WithMemoryAccessOverride(raw => raw == 0 ? (UInt128)0 : (UInt128)1024);

        var function = CreateLargeConstEmbeddingFunction(out var sequenceLength);
        var result = await RunAutoDistributedAsync(function, "const_embed_out", 64L * 1024L * 1024L);
        var placement = Placement();
        var hiddenShard = LargeHiddenShardOutputType(sequenceLength, placement);
        var sequenceShard = LargeSequenceShardOutputType(sequenceLength, placement);

        var gather = Assert.Single(FindCalls(result.Body, call => call.Target is IR.Tensors.Gather));
        Assert.Equal(LargeHiddenShardWeightType(placement), Assert.IsType<DistributedType>(gather.Arguments[0].CheckedType));
        Assert.Equal(SmallBroadcastInputIdsType(sequenceLength, placement), Assert.IsType<DistributedType>(gather.Arguments[1].CheckedType));
        Assert.Equal(hiddenShard, Assert.IsType<DistributedType>(gather.CheckedType));
        AssertHasBoxing(result.Body, hiddenShard, sequenceShard);
        AssertDoesNotHaveBoxing(result.Body, LargeBroadcastOutputType(sequenceLength, placement), sequenceShard);
    }

    [Fact]
    public async Task TestCudaMemoryCapAvoidsDynamicSequenceSplitMatMulWithBroadcastStaticWeight()
    {
        using var memoryAccessOverride = CostUtility.WithMemoryAccessOverride(raw => raw == 0 ? (UInt128)0 : (UInt128)1024);

        var function = CreateConstMatMulFunction(out var sequenceLength);
        var result = await RunAutoDistributedAsync(function, "matmul_out", 589824, [SBP.B, SBP.S(0)]);
        var placement = Placement();
        var badLhs = MatMulSequenceShardLhsType(sequenceLength, placement);
        var badRhs = MatMulBroadcastWeightType(placement);

        Assert.Empty(FindCalls(result.Body, call =>
            call.Target is IR.Math.MatMul
            && call.Arguments[0].CheckedType == badLhs
            && call.Arguments[1].CheckedType == badRhs));
    }

    private Function CreateEmbeddingFunction(out DimVar sequenceLength)
    {
        sequenceLength = new DimVar("sequence_length") { Metadata = { Range = (1, 64) } };
        var inputIds = new Var("input_ids", new TensorType(DataTypes.Int64, [sequenceLength]));
        var weight = new Var("embed_tokens_weight", new TensorType(DataTypes.Float16, [151936, 1024]));
        var gather = IR.F.Tensors.Gather(weight, 0, inputIds);
        var output = IR.F.Math.Unary(UnaryOp.Abs, gather);
        output.Metadata.OutputNames = [OutputName];
        return new Function("main", output, [inputIds, weight]);
    }

    private Function CreateConstEmbeddingFunction(out DimVar sequenceLength)
    {
        sequenceLength = new DimVar("sequence_length") { Metadata = { Range = (1, 64) } };
        var inputIds = new Var("input_ids", new TensorType(DataTypes.Int64, [sequenceLength]));
        Expr weight = Tensor.From<float>(Enumerable.Repeat(0f, 1024 * 256).ToArray(), [1024, 256]);
        var gather = IR.F.Tensors.Gather(weight, 0, inputIds);
        var output = IR.F.Math.Unary(UnaryOp.Abs, gather);
        output.Metadata.OutputNames = ["const_embed_out"];
        return new Function("main", output, [inputIds]);
    }

    private Function CreateLargeConstEmbeddingFunction(out DimVar sequenceLength)
    {
        sequenceLength = new DimVar("sequence_length") { Metadata = { Range = (1, 64) } };
        var inputIds = new Var("input_ids", new TensorType(DataTypes.Int64, [sequenceLength]));
        Expr weight = Tensor.From<float>(Enumerable.Repeat(0f, 2048 * 1024).ToArray(), [2048, 1024]);
        var gather = IR.F.Tensors.Gather(weight, 0, inputIds);
        var output = IR.F.Math.Unary(UnaryOp.Abs, gather);
        output.Metadata.OutputNames = ["const_embed_out"];
        return new Function("main", output, [inputIds]);
    }

    private Function CreateConstMatMulFunction(out DimVar sequenceLength)
    {
        sequenceLength = new DimVar("sequence_length") { Metadata = { Range = (1, 64) } };
        var input = new Var("hidden_states", new TensorType(DataTypes.Float16, [sequenceLength, 1024]));
        Expr weight = Tensor.From<Half>(Enumerable.Repeat((Half)0f, 1024 * 2048).ToArray(), [1024, 2048]);
        var matmul = IR.F.Math.MatMul(input, weight);
        matmul.Metadata.OutputNames = ["matmul_out"];
        return new Function("main", matmul, [input]);
    }

    private async Task<(Function Result, DimVar SequenceLength)> RunAutoDistributedAsync(long? cudaPeMemoryLimitBytes = null)
    {
        var function = CreateEmbeddingFunction(out var sequenceLength);
        var result = await RunAutoDistributedAsync(function, OutputName, cudaPeMemoryLimitBytes);
        return (result, sequenceLength);
    }

    private async Task<Function> RunAutoDistributedAsync(Function function, string outputName, long? cudaPeMemoryLimitBytes = null, IReadOnlyList<SBP>? outputSbps = null)
    {
        var schemePath = WriteTemporaryDistributedScheme(outputName, outputSbps);
        try
        {
            var memoryCapacities = cudaPeMemoryLimitBytes is { } limitBytes
                ? new[] { 524288, checked((int)limitBytes) }
                : new[] { 524288, int.MaxValue };
            CompileOptions.TargetOptions = new NTTTargetOptions()
            {
                Hierarchies = [[16]],
                HierarchyNames = HierarchyName,
                DistributedScheme = schemePath,
                MemoryCapacities = memoryCapacities,
            };

            var pass = new AutoDistributedPass(false, CUDATarget.Kind, CompileOptions);
            var result = await pass.RunAsync(function, new());
            return Assert.IsType<Function>(result);
        }
        finally
        {
            File.Delete(schemePath);
        }
    }

    private string WriteTemporaryDistributedScheme(string outputName = OutputName, IReadOnlyList<SBP>? outputSbps = null)
    {
        var scheme = new DistributedSchema(
            "1",
            "qwen3_embedding",
            [new DistributedSchema.Node(outputName, outputSbps?.ToArray() ?? [SBP.S(0), SBP.B], [16], HierarchyName)]);
        var options = new JsonSerializerOptions { WriteIndented = true };
        options.Converters.Add(new SBPConverter());
        var path = Path.Combine(Path.GetTempPath(), $"nncase_qwen_embedding_{Guid.NewGuid():N}.json");
        File.WriteAllText(path, JsonSerializer.Serialize(scheme, options));
        return path;
    }

    private Placement Placement() => new([16], HierarchyName);

    private DistributedType BroadcastInputIdsType(DimVar sequenceLength, Placement placement) =>
        new(new TensorType(DataTypes.Int64, [sequenceLength]), [SBP.B], placement);

    private DistributedType BroadcastWeightType(Placement placement) =>
        new(new TensorType(DataTypes.Float16, [151936, 1024]), [SBP.B, SBP.B], placement);

    private DistributedType HiddenShardWeightType(Placement placement) =>
        new(new TensorType(DataTypes.Float16, [151936, 1024]), [SBP.B, SBP.S(0)], placement);

    private DistributedType BroadcastOutputType(DimVar sequenceLength, Placement placement) =>
        new(new TensorType(DataTypes.Float16, [sequenceLength, 1024]), [SBP.B, SBP.B], placement);

    private DistributedType HiddenShardOutputType(DimVar sequenceLength, Placement placement) =>
        new(new TensorType(DataTypes.Float16, [sequenceLength, 1024]), [SBP.B, SBP.S(0)], placement);

    private DistributedType SequenceShardOutputType(DimVar sequenceLength, Placement placement) =>
        new(new TensorType(DataTypes.Float16, [sequenceLength, 1024]), [SBP.S(0), SBP.B], placement);

    private DistributedType SmallBroadcastInputIdsType(DimVar sequenceLength, Placement placement) =>
        new(new TensorType(DataTypes.Int64, [sequenceLength]), [SBP.B], placement);

    private DistributedType SmallHiddenShardWeightType(Placement placement) =>
        new(new TensorType(DataTypes.Float32, [1024, 256]), [SBP.B, SBP.S(0)], placement);

    private DistributedType SmallHiddenShardOutputType(DimVar sequenceLength, Placement placement) =>
        new(new TensorType(DataTypes.Float32, [sequenceLength, 256]), [SBP.B, SBP.S(0)], placement);

    private DistributedType SmallBroadcastOutputType(DimVar sequenceLength, Placement placement) =>
        new(new TensorType(DataTypes.Float32, [sequenceLength, 256]), [SBP.B, SBP.B], placement);

    private DistributedType SmallSequenceShardOutputType(DimVar sequenceLength, Placement placement) =>
        new(new TensorType(DataTypes.Float32, [sequenceLength, 256]), [SBP.S(0), SBP.B], placement);

    private DistributedType LargeHiddenShardWeightType(Placement placement) =>
        new(new TensorType(DataTypes.Float32, [2048, 1024]), [SBP.B, SBP.S(0)], placement);

    private DistributedType LargeHiddenShardOutputType(DimVar sequenceLength, Placement placement) =>
        new(new TensorType(DataTypes.Float32, [sequenceLength, 1024]), [SBP.B, SBP.S(0)], placement);

    private DistributedType LargeBroadcastOutputType(DimVar sequenceLength, Placement placement) =>
        new(new TensorType(DataTypes.Float32, [sequenceLength, 1024]), [SBP.B, SBP.B], placement);

    private DistributedType LargeSequenceShardOutputType(DimVar sequenceLength, Placement placement) =>
        new(new TensorType(DataTypes.Float32, [sequenceLength, 1024]), [SBP.S(0), SBP.B], placement);

    private DistributedType MatMulSequenceShardLhsType(DimVar sequenceLength, Placement placement) =>
        new(new TensorType(DataTypes.Float16, [sequenceLength, 1024]), [SBP.S(0), SBP.B], placement);

    private DistributedType MatMulBroadcastWeightType(Placement placement) =>
        new(new TensorType(DataTypes.Float16, [1024, 2048]), [SBP.B, SBP.B], placement);

    private void AssertHasBoxing(BaseExpr root, DistributedType inputType, DistributedType outputType)
    {
        Assert.NotEmpty(FindCalls(root, call => IsBoxing(call, inputType, outputType)));
    }

    private void AssertDoesNotHaveBoxing(BaseExpr root, DistributedType inputType, DistributedType outputType)
    {
        Assert.Empty(FindCalls(root, call => IsBoxing(call, inputType, outputType)));
    }

    private bool IsBoxing(Call call, DistributedType inputType, DistributedType outputType)
    {
        return call.Target is IR.Distributed.Boxing boxing
            && boxing.NewType == outputType
            && call.Arguments[0].CheckedType == inputType
            && call.CheckedType == outputType;
    }

    private IReadOnlyList<Call> FindCalls(BaseExpr root, Func<Call, bool> predicate)
    {
        var calls = new List<Call>();
        var stack = new Stack<BaseExpr>();
        var seen = new HashSet<BaseExpr>(new ReferenceEqualityComparer<BaseExpr>());
        stack.Push(root);

        while (stack.Count != 0)
        {
            var current = stack.Pop();
            if (!seen.Add(current))
            {
                continue;
            }

            if (current is Call call && predicate(call))
            {
                calls.Add(call);
            }

            foreach (var operand in current.Operands)
            {
                stack.Push(operand);
            }
        }

        return calls;
    }
}
