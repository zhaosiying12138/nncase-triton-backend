// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.CommandLine;
using System.CommandLine.Invocation;
using Nncase.CodeGen;
using Nncase.CodeGen.NTT.CUDA;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Passes;

namespace Nncase.Targets;

/// <summary>
/// Minimal CUDA target backed by the NTT pipeline shape.
/// </summary>
public sealed class CUDATarget : Target
{
    public const string Kind = "cuda";

    private readonly CUDAModuleCompiler _cudaModuleCompiler = new();

    public CUDATarget()
    {
        ModuleCompilers = [_cudaModuleCompiler];
    }

    public override string Name => Kind;

    public override IReadOnlyList<IModuleCompiler> ModuleCompilers { get; }

    public override bool EnableAutoVectorize => false;

    public override bool EnableAutoPacking => false;

    public override (Command Command, Func<InvocationContext, Command, ITargetOptions> Parser) RegisterCommandAndParser()
    {
        var cmd = new NTTTargetOptionsCommand(Kind);

        ITargetOptions ParseTargetCompileOptions(InvocationContext context, Command command)
        {
            var binder = new NTTTargetOptionsBinder(cmd);
            return binder.GetBoundValue(context);
        }

        return (cmd, ParseTargetCompileOptions);
    }

    public override void RegisterTargetDependentPass(IPassManager passManager, CompileOptions options)
    {
        EnsureNTTTargetOptions(options);
    }

    public override void RegisterAffineSelectionPass(IPassManager passManager, CompileOptions options)
    {
        EnsureNTTTargetOptions(options);
        passManager.Add<NTTAffineSelectionPass>(Kind);
    }

    public override void RegisterTIRSelectionPass(IPassManager passManager, CompileOptions options)
    {
        EnsureNTTTargetOptions(options);
        passManager.Add<NTTTIRSelectionPass>(Kind);
    }

    private static void EnsureNTTTargetOptions(CompileOptions options)
    {
        if (options.TargetOptions is null)
        {
            options.TargetOptions = new NTTTargetOptions();
        }
        else if (options.TargetOptions is not INTTTargetOptions)
        {
            throw new NotSupportedException("CUDA target expects NTT target options.");
        }
    }
}

public sealed class CUDAModuleCompiler : IModuleCompiler
{
    public string ModuleKind => CUDATarget.Kind;

    public MaskVectorStyle MaskVectorStyle => MaskVectorStyle.Fat;

    public static bool IsSupportedPagedAttentionConfig(IPagedAttentionConfig config)
    {
        return config.VectorizedAxes.Count == 0 && config.Lanes.Count == 0;
    }

    public static void ThrowIfUnsupportedPagedAttentionConfig(IPagedAttentionConfig config)
    {
        if (!IsSupportedPagedAttentionConfig(config))
        {
            throw new NotSupportedException("CUDA target v1 does not support vectorized PagedAttention KV cache layouts.");
        }
    }

    public IModuleBuilder CreateModuleBuilder(CompileOptions options) => new CudaModuleBuilder(options);

    public bool IsSupportedCall(Call call, CompileOptions options)
    {
        if (call.Target is not Op op)
        {
            return false;
        }

        foreach (var config in GetPagedAttentionConfigs(call))
        {
            ThrowIfUnsupportedPagedAttentionConfig(config);
        }

        return PassUtility.IsCpuSupported(op, call, call.Arguments, ModuleKind);
    }

    private static IReadOnlyList<IPagedAttentionConfig> GetPagedAttentionConfigs(Call call)
    {
        var configs = new List<IPagedAttentionConfig>();
        if (call.Target is CreatePagedAttentionKVCache createPagedAttentionKVCache)
        {
            configs.Add(createPagedAttentionKVCache.Config);
        }
        else if (call.Target is TIR.NTT.CreatePagedAttentionKVCache createPagedAttentionKVCacheTir)
        {
            configs.Add(createPagedAttentionKVCacheTir.Config);
        }

        foreach (var argument in call.Arguments.ToArray())
        {
            if (TryGetPagedAttentionConfig(argument.CheckedType, out var config))
            {
                configs.Add(config);
            }
        }

        return configs;
    }

    private static bool TryGetPagedAttentionConfig(IRType type, out IPagedAttentionConfig config)
    {
        config = type switch
        {
            TensorType { DType: ReferenceType { ElemType: PagedAttentionKVCacheType kvCacheType } }
                when kvCacheType.Config is not null => kvCacheType.Config,
            DistributedType { TensorType: TensorType { DType: ReferenceType { ElemType: PagedAttentionKVCacheType kvCacheType } } }
                when kvCacheType.Config is not null => kvCacheType.Config,
            _ => null!,
        };

        return config is not null;
    }
}
