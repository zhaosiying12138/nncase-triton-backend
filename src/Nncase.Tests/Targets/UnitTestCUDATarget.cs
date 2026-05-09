// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Passes;
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
}
