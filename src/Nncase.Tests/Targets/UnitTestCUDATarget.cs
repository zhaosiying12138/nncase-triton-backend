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
        var target = CompilerServices.GetTarget(CUDATarget.Kind);
        var function = BuildFlashAttentionFunction();

        CompileOptions.TargetOptions = new NTTTargetOptions { FusedKernelMode = CudaFusedKernelMode.Off };
        var offModule = new IRModule(function);
        var offPasses = CompileSession.CreatePassManager("cuda_fused_kernel_off");
        target.RegisterTargetDependentPass(offPasses, CompileOptions);
        await offPasses.RunAsync(offModule);
        Assert.DoesNotContain(ExprCollector.Collect(offModule.Entry!), e => e is Call { Target: Fusion });

        CompileOptions.TargetOptions = new NTTTargetOptions { FusedKernelMode = CudaFusedKernelMode.Compute };
        var computeModule = new IRModule(function);
        var computePasses = CompileSession.CreatePassManager("cuda_fused_kernel_compute");
        target.RegisterTargetDependentPass(computePasses, CompileOptions);
        await computePasses.RunAsync(computeModule);

        var fusionCall = ExprCollector.Collect(computeModule.Entry!)
            .OfType<Call>()
            .Single(call => call.Target is Fusion);
        var fusion = Assert.IsType<Fusion>(fusionCall.Target);
        Assert.Equal(CUDATarget.Kind, fusion.ModuleKind);
        Assert.StartsWith("cuda.flash_attention", fusion.Name, StringComparison.Ordinal);

        var autoDistributedPass = new AutoDistributedPass(false, CUDATarget.Kind, CompileOptions);
        var distributed = await autoDistributedPass.RunAsync(computeModule.Entry!, new());
        var distributedFusionCall = ExprCollector.Collect(distributed)
            .OfType<Call>()
            .Single(call => call.Target is Fusion);
        Assert.StartsWith("cuda.flash_attention", ((Fusion)distributedFusionCall.Target).Name, StringComparison.Ordinal);

        var tirPass = new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind);
        var prim = Assert.IsType<PrimFunction>(await tirPass.RunAsync(distributed, new()));
        var fusedKernelCall = ExprCollector.Collect(prim)
            .OfType<Call>()
            .Single(call => call.Target is TIR.NTT.FusedKernel);
        var fusedKernel = Assert.IsType<TIR.NTT.FusedKernel>(fusedKernelCall.Target);
        Assert.StartsWith("cuda.flash_attention", fusedKernel.FusionName, StringComparison.Ordinal);
        Assert.Equal(4, fusedKernelCall.Arguments.Length);

        var collectorType = typeof(TritonPythonSourceBuilder).Assembly.GetType("Nncase.CodeGen.NTT.CUDA.CudaTritonLaunchCollector");
        Assert.NotNull(collectorType);
        var collector = Activator.CreateInstance(collectorType)!;
        var launches = Assert.IsAssignableFrom<IReadOnlyList<CudaTritonKernelLaunch>>(
            collectorType.GetMethod("Collect")!.Invoke(collector, [prim]));
        var launch = launches.Single(l => l.OpName.StartsWith("fusion.cuda.flash_attention", StringComparison.Ordinal));
        Assert.NotNull(launch.BufferArguments);
        Assert.Equal("Input", launch.BufferArguments!["arg0"].Memory.Location);
        Assert.Equal("Input", launch.BufferArguments["arg1"].Memory.Location);
        Assert.Equal("Input", launch.BufferArguments["arg2"].Memory.Location);
        Assert.Equal("Output", launch.BufferArguments["arg3"].Memory.Location);
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
}
