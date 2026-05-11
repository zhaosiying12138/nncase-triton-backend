// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.CodeGen.NTT.CUDA;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Passes;
using Nncase.Passes.Distributed;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TargetTest;

[Collection(nameof(NotThreadSafeResourceCollection))]
[AutoSetupTestMethod(InitSession = true)]
public class UnitTestCUDATarget : TestClassBase
{
    public override string DefaultTargetName { get; set; } = CUDATarget.Kind;

    [Fact]
    public void TestCUDATargetRegistrationAndCapabilities()
    {
        var target = CompilerServices.GetTarget(CUDATarget.Kind);

        Assert.Equal("cuda", target.Name);
        Assert.False(target.EnableAutoVectorize);
        Assert.False(target.EnableAutoPacking);
        Assert.Single(target.ModuleCompilers);
        Assert.Equal(CUDATarget.Kind, target.ModuleCompilers.Single().ModuleKind);
    }

    [Fact]
    public void TestCPUTargetKeepsAutoVectorizeAndPackingEnabled()
    {
        var target = CompilerServices.GetTarget(CPUTarget.Kind);

        Assert.True(target.EnableAutoVectorize);
        Assert.True(target.EnableAutoPacking);
    }

    [Fact]
    public void TestPrimFunctionWrapperPreservesCudaModuleKind()
    {
        var primFunction = new PrimFunction("cuda_kernel", CUDATarget.Kind, Sequential.Empty, Array.Empty<IVar>());
        var wrapper = new PrimFunctionWrapper("cuda_kernel_wrapper", primFunction, 0);

        Assert.Equal(CUDATarget.Kind, wrapper.ModuleKind);

        var retargeted = (PrimFunctionWrapper)((BaseFunction)wrapper).With(moduleKind: "other");
        Assert.Equal("other", retargeted.ModuleKind);
    }

    [Fact]
    public void TestCUDAPagedAttentionRejectsVectorizedKVCache()
    {
        var config = new PagedAttentionConfig(
            1,
            2,
            64,
            DataTypes.Float16,
            16,
            new[]
            {
                PagedKVCacheDimKind.NumBlocks,
                PagedKVCacheDimKind.NumLayers,
                PagedKVCacheDimKind.KV,
                PagedKVCacheDimKind.BlockSize,
                PagedKVCacheDimKind.NumKVHeads,
                PagedKVCacheDimKind.HeadDim,
            },
            new[] { PagedKVCacheDimKind.HeadDim },
            new[] { 32 },
            Array.Empty<PagedKVCacheDimKind>(),
            Array.Empty<SBPSplit>());

        var ex = Assert.Throws<NotSupportedException>(() =>
            CUDAModuleCompiler.ThrowIfUnsupportedPagedAttentionConfig(config));
        Assert.Contains("vectorized", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task TestCUDATIRSelectionSupportsUnvectorizedRoPE()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, [1, 2, 4]));
        var cos = new Var("cos", new TensorType(DataTypes.Float32, [2, 4]));
        var sin = new Var("sin", new TensorType(DataTypes.Float32, [2, 4]));
        var body = IR.F.NN.RoPE(input, cos, sin);
        var function = new Function("main", CUDATarget.Kind, body, [input, cos, sin]);
        var pass = new NTTTIRSelectionPass(new CompileOptions(), CUDATarget.Kind);

        var post = await pass.RunAsync(function, new());
        var prim = Assert.IsType<PrimFunction>(post);

        Assert.Contains(ExprCollector.Collect(prim), e => e is Call { Target: TIR.NTT.RoPE });
    }

    [Fact]
    public async Task TestCudaComputeFusedKernelPassFusesAttentionToCudaFusion()
    {
        await AssertCudaComputeFusionAsync(BuildFlashAttentionFunction(), "cuda.flash_attention", 4);
    }

    [Fact]
    public async Task TestCudaComputeFusedKernelPassFusesSwishMulToCudaFusion()
    {
        await AssertCudaComputeFusionAsync(BuildSwishMulFunction(), "cuda.swish_mul", 3);
    }

    [Fact]
    public async Task TestCudaComputeFusedKernelPassFusesMatmulSwishMulToCudaFusion()
    {
        await AssertCudaComputeFusionAsync(BuildMatmulSwishMulFunction(), "cuda.matmul_swish_mul", 4);
    }

    [Fact]
    public async Task TestCudaComputeFusedKernelPassFusesSiluMulMatmulToCudaFusion()
    {
        await AssertCudaComputeFusionAsync(BuildSiluMulMatmulFunction(), "cuda.silu_mul_matmul", 4);
    }

    [Fact]
    public async Task TestCudaComputeFusedKernelPassFusesMatmulMulMatmulToCudaFusion()
    {
        await AssertCudaComputeFusionAsync(BuildMatmulMulMatmulFunction(), "cuda.matmul_mul_matmul", 5);
    }

    [Fact]
    public async Task TestCudaComputeFusedKernelPassFusesFullMlpToCudaFusion()
    {
        await AssertCudaComputeFusionAsync(BuildFullMlpFunction(), "cuda.matmul_silu_matmul_mul_matmul", 5);
    }

    [Fact]
    public async Task TestCudaComputeFusedKernelPassFusesLayerNormTransposeToCudaFusion()
    {
        await AssertCudaComputeFusionAsync(BuildLayerNormTransposeFunction(), "cuda.layer_norm_transpose", 4);
    }

    [Fact]
    public async Task TestCudaComputeFusedKernelPassFusesLayerNormMatmulToCudaFusion()
    {
        await AssertCudaComputeFusionAsync(BuildLayerNormMatmulFunction(), "cuda.layer_norm_matmul", 5);
    }

    [Fact]
    public async Task TestCudaComputeFusedKernelPassFusesMulCosToCudaFusion()
    {
        await AssertCudaComputeFusionAsync(BuildMulCosFunction(), "cuda.mul_cos", 3);
    }

    [Fact]
    public async Task TestCudaComputeFusedKernelPassFusesMulSinToCudaFusion()
    {
        await AssertCudaComputeFusionAsync(BuildMulSinFunction(), "cuda.mul_sin", 3);
    }

    [Fact]
    public async Task TestCudaComputeFusedKernelPassFusesRoPEToCudaFusion()
    {
        await AssertCudaComputeFusionAsync(BuildRoPEFunction(), "cuda.rope", 4);
    }

    [Fact]
    public async Task TestCudaComputeFusionRunsAfterAutoDistributedAndSurvivesAutoTiling()
    {
        CompileOptions.TargetOptions = new NTTTargetOptions { FusedKernelMode = CudaFusedKernelMode.Compute };
        var module = new IRModule(BuildFullMlpFunction());
        var compiler = (Nncase.Compiler.Compiler)CompileSession.Compiler;

        var targetPasses = CompileSession.CreatePassManager("cuda_compute_target_dependent_no_fusion");
        CompileSession.Target.RegisterTargetDependentPass(targetPasses, CompileOptions);
        await targetPasses.RunAsync(module);
        Assert.DoesNotContain(ExprCollector.Collect(module.Entry!), e => e is Call { Target: Fusion });

        var distributedPasses = CompileSession.CreatePassManager("cuda_compute_autodistributed_no_fusion");
        compiler.AutoDistributedPass(distributedPasses);
        await distributedPasses.RunAsync(module);
        Assert.DoesNotContain(ExprCollector.Collect(module.Entry!), e => e is Call { Target: Fusion });

        var tilingPasses = CompileSession.CreatePassManager("cuda_compute_autotiling_after_fusion");
        compiler.AutoTilingPass(tilingPasses);
        await tilingPasses.RunAsync(module);
        Assert.Contains(
            ExprCollector.Collect(module.Entry!),
            e => e is Call { Target: Fusion fusion } &&
                 fusion.Name.StartsWith("cuda.matmul_silu_matmul_mul_matmul", StringComparison.Ordinal));

        var tirPasses = CompileSession.CreatePassManager("cuda_compute_tir_after_autotiling");
        compiler.TIRPass(tirPasses);
        await tirPasses.RunAsync(module);
        Assert.Contains(
            ExprCollector.Collect(module.Entry!),
            e => e is Call { Target: TIR.NTT.FusedKernel fused } &&
                 fused.FusionName.StartsWith("cuda.matmul_silu_matmul_mul_matmul", StringComparison.Ordinal));
    }

    private static Function BuildFlashAttentionFunction()
    {
        var q = new Var("q", new TensorType(DataTypes.Float32, [2, 4, 8]));
        var k = new Var("k", new TensorType(DataTypes.Float32, [2, 4, 8]));
        var v = new Var("v", new TensorType(DataTypes.Float32, [2, 4, 8]));
        var kT = IR.F.Tensors.Transpose(k, [0, 2, 1]);
        var scores = IR.F.Math.MatMul(q, kT);
        var scaled = IR.F.Math.Mul(scores, 0.35355338f);
        var probs = IR.F.NN.Softmax(scaled, -1);
        var output = IR.F.Math.MatMul(probs, v);
        return new Function("main", CUDATarget.Kind, output, [q, k, v]);
    }

    private static Function BuildSwishMulFunction()
    {
        var gate = new Var("gate", new TensorType(DataTypes.Float32, [2, 4]));
        var up = new Var("up", new TensorType(DataTypes.Float32, [2, 4]));
        var body = IR.F.Math.Mul(IR.F.NN.Swish(gate), up);
        return new Function("main", CUDATarget.Kind, body, [gate, up]);
    }

    private static Function BuildMatmulSwishMulFunction()
    {
        var gate = new Var("gate", new TensorType(DataTypes.Float32, [2, 4]));
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, [2, 8]));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, [8, 4]));
        var up = IR.F.Math.MatMul(lhs, rhs);
        var body = IR.F.Math.Mul(IR.F.NN.Swish(gate), up);
        return new Function("main", CUDATarget.Kind, body, [gate, lhs, rhs]);
    }

    private static Function BuildSiluMulMatmulFunction()
    {
        var gate = new Var("gate", new TensorType(DataTypes.Float32, [2, 8]));
        var up = new Var("up", new TensorType(DataTypes.Float32, [2, 8]));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, [8, 4]));
        var siluUp = IR.F.Math.Mul(IR.F.NN.Swish(gate), up);
        var body = IR.F.Math.MatMul(siluUp, rhs);
        return new Function("main", CUDATarget.Kind, body, [gate, up, rhs]);
    }

    private static Function BuildMatmulMulMatmulFunction()
    {
        var gate = new Var("gate", new TensorType(DataTypes.Float32, [2, 8]));
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, [2, 4]));
        var upRhs = new Var("up_rhs", new TensorType(DataTypes.Float32, [4, 8]));
        var downRhs = new Var("down_rhs", new TensorType(DataTypes.Float32, [8, 3]));
        var up = IR.F.Math.MatMul(lhs, upRhs);
        var gated = IR.F.Math.Mul(gate, up);
        var body = IR.F.Math.MatMul(gated, downRhs);
        return new Function("main", CUDATarget.Kind, body, [gate, lhs, upRhs, downRhs]);
    }

    private static Function BuildFullMlpFunction()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, [2, 4]));
        var gateRhs = new Var("gate_rhs", new TensorType(DataTypes.Float32, [4, 8]));
        var upRhs = new Var("up_rhs", new TensorType(DataTypes.Float32, [4, 8]));
        var downRhs = new Var("down_rhs", new TensorType(DataTypes.Float32, [8, 3]));
        var gate = IR.F.Math.MatMul(lhs, gateRhs);
        var up = IR.F.Math.MatMul(lhs, upRhs);
        var hidden = IR.F.Math.Mul(IR.F.NN.Swish(gate), up);
        var body = IR.F.Math.MatMul(hidden, downRhs);
        return new Function("main", CUDATarget.Kind, body, [lhs, gateRhs, upRhs, downRhs]);
    }

    private static Function BuildLayerNormTransposeFunction()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, [2, 4]));
        var scale = new Var("scale", new TensorType(DataTypes.Float32, [4]));
        var bias = new Var("bias", new TensorType(DataTypes.Float32, [4]));
        var normalized = IR.F.NN.LayerNorm(-1, 1e-5f, input, scale, bias);
        var body = IR.F.Tensors.Transpose(normalized, [1, 0]);
        return new Function("main", CUDATarget.Kind, body, [input, scale, bias]);
    }

    private static Function BuildLayerNormMatmulFunction()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, [2, 4]));
        var scale = new Var("scale", new TensorType(DataTypes.Float32, [4]));
        var bias = new Var("bias", new TensorType(DataTypes.Float32, [4]));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, [4, 3]));
        var normalized = IR.F.NN.LayerNorm(-1, 1e-5f, input, scale, bias);
        var body = IR.F.Math.MatMul(normalized, rhs);
        return new Function("main", CUDATarget.Kind, body, [input, scale, bias, rhs]);
    }

    private static Function BuildMulCosFunction()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, [2, 4]));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, [2, 4]));
        var body = IR.F.Math.Cos(IR.F.Math.Mul(lhs, rhs));
        return new Function("main", CUDATarget.Kind, body, [lhs, rhs]);
    }

    private static Function BuildMulSinFunction()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, [2, 4]));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, [2, 4]));
        var body = IR.F.Math.Sin(IR.F.Math.Mul(lhs, rhs));
        return new Function("main", CUDATarget.Kind, body, [lhs, rhs]);
    }

    private static Function BuildRoPEFunction()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, [1, 2, 4]));
        var cos = new Var("cos", new TensorType(DataTypes.Float32, [2, 4]));
        var sin = new Var("sin", new TensorType(DataTypes.Float32, [2, 4]));
        var body = IR.F.NN.RoPE(input, cos, sin);
        return new Function("main", CUDATarget.Kind, body, [input, cos, sin]);
    }

    private async Task AssertCudaComputeFusionAsync(Function function, string expectedFusionPrefix, int expectedArgumentCount)
    {
        var target = CompilerServices.GetTarget(CUDATarget.Kind);

        CompileOptions.TargetOptions = new NTTTargetOptions { FusedKernelMode = CudaFusedKernelMode.Off };
        var offModule = new IRModule(function);
        var offPasses = CompileSession.CreatePassManager($"cuda_fused_kernel_off_{expectedFusionPrefix}");
        target.RegisterTargetDependentPass(offPasses, CompileOptions);
        await offPasses.RunAsync(offModule);
        Assert.DoesNotContain(ExprCollector.Collect(offModule.Entry!), e => e is Call { Target: Fusion });

        CompileOptions.TargetOptions = new NTTTargetOptions { FusedKernelMode = CudaFusedKernelMode.Compute };
        var computeModule = new IRModule(function);
        var targetPasses = CompileSession.CreatePassManager($"cuda_fused_kernel_compute_target_{expectedFusionPrefix}");
        target.RegisterTargetDependentPass(targetPasses, CompileOptions);
        await targetPasses.RunAsync(computeModule);
        Assert.DoesNotContain(ExprCollector.Collect(computeModule.Entry!), e => e is Call { Target: Fusion });

        var autoDistributedPass = new AutoDistributedPass(false, CUDATarget.Kind, CompileOptions);
        var distributed = await autoDistributedPass.RunAsync(computeModule.Entry!, new());
        Assert.DoesNotContain(ExprCollector.Collect(distributed), e => e is Call { Target: Fusion });

        computeModule = new IRModule((Function)distributed);
        var computePasses = CompileSession.CreatePassManager($"cuda_fused_kernel_compute_affine_{expectedFusionPrefix}");
        target.RegisterAffineSelectionPass(computePasses, CompileOptions);
        await computePasses.RunAsync(computeModule);

        var fusionCall = ExprCollector.Collect(computeModule.Entry!)
            .OfType<Call>()
            .Single(call => call.Target is Fusion fusion && fusion.Name.StartsWith(expectedFusionPrefix, StringComparison.Ordinal));
        var fusion = Assert.IsType<Fusion>(fusionCall.Target);
        Assert.Equal(CUDATarget.Kind, fusion.ModuleKind);

        var tirPass = new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind);
        var prim = Assert.IsType<PrimFunction>(await tirPass.RunAsync(computeModule.Entry!, new()));
        var fusedKernelCall = ExprCollector.Collect(prim)
            .OfType<Call>()
            .Single(call => call.Target is TIR.NTT.FusedKernel fused && fused.FusionName.StartsWith(expectedFusionPrefix, StringComparison.Ordinal));
        var fusedKernel = Assert.IsType<TIR.NTT.FusedKernel>(fusedKernelCall.Target);
        Assert.StartsWith(expectedFusionPrefix, fusedKernel.FusionName, StringComparison.Ordinal);
        Assert.Equal(expectedArgumentCount, fusedKernelCall.Arguments.Length);

        var collectorType = typeof(TritonPythonSourceBuilder).Assembly.GetType("Nncase.CodeGen.NTT.CUDA.CudaTritonLaunchCollector");
        Assert.NotNull(collectorType);
        var collector = Activator.CreateInstance(collectorType)!;
        var launches = Assert.IsAssignableFrom<IReadOnlyList<CudaTritonKernelLaunch>>(
            collectorType.GetMethod("Collect")!.Invoke(collector, [prim]));
        var launch = launches.Single(l => l.OpName.StartsWith($"fusion.{expectedFusionPrefix}", StringComparison.Ordinal));
        Assert.NotNull(launch.BufferArguments);
        Assert.Contains(launch.BufferArguments!["arg0"].Memory.Location, new[] { "Input", "Data" });
        Assert.Contains(launch.BufferArguments[$"arg{expectedArgumentCount - 1}"].Memory.Location, new[] { "Output", "Data" });
    }
}
