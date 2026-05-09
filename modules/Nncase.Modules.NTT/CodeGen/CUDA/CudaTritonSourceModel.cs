// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CodeGen.NTT.CUDA;

public enum CudaTritonLaunchKind
{
    Compute,
    Memcopy,
    Boxing,
    Reduce,
    Matmul,
    Collective,
    Function,
}

public sealed record CudaTritonKernelLaunch(
    int Ordinal,
    CudaTritonLaunchKind Kind,
    string OpName,
    IReadOnlyList<string> Arguments,
    bool RequiresCollective,
    IReadOnlyDictionary<string, CudaTritonBufferDesc>? BufferArguments = null,
    IReadOnlyDictionary<string, object?>? OpAttributes = null);

public sealed record CudaTritonDimDesc(
    string Kind,
    long? Value = null,
    string? Symbol = null,
    string? Expression = null);

public sealed record CudaTritonStrideDesc(
    string Kind,
    long? Value = null,
    string? Symbol = null,
    string? Expression = null);

public sealed record CudaTritonMemoryDesc(
    string Location,
    int Hierarchy,
    int Alignment,
    CudaTritonDimDesc BufferSizeBytes,
    CudaTritonDimDesc SpanStartBytes,
    CudaTritonDimDesc SpanSizeBytes,
    string? BaseStart = null);

public sealed record CudaTritonBufferDesc(
    string Name,
    string DType,
    int ElementSizeBytes,
    int Rank,
    IReadOnlyList<CudaTritonDimDesc> Shape,
    IReadOnlyList<CudaTritonStrideDesc> Strides,
    CudaTritonMemoryDesc Memory,
    string? DistributedType = null);

public sealed record CudaTritonReturnDesc(
    int Index,
    string Value,
    CudaTritonBufferDesc Buffer);

public sealed record CudaTritonFunctionParameterDesc(
    string Name,
    string? Type = null,
    string? Buffer = null);

public sealed record CudaTritonFunctionSource(
    uint Id,
    string Name,
    bool IsEntry,
    IReadOnlyList<string> Parameters,
    ulong LocalDataPoolSize,
    ulong OutputPoolSize,
    ulong RdataPoolSize,
    IReadOnlyList<CudaTritonKernelLaunch> Launches,
    ulong BlockLocalDataPoolSize = 0,
    IReadOnlyDictionary<string, CudaTritonBufferDesc>? Buffers = null,
    IReadOnlyList<CudaTritonFunctionParameterDesc>? ParameterDescs = null,
    IReadOnlyList<CudaTritonReturnDesc>? Returns = null);

public sealed record CudaTritonModuleSource(
    int PeCount,
    ulong RdataPoolSize,
    ulong ThreadLocalRdataPoolSize,
    ulong BlockLocalRdataPoolSize,
    IReadOnlyList<CudaTritonFunctionSource> Functions);

internal static class CudaSectionNames
{
    public const string CudaMeta = ".cuda.meta";

    public const string TritonModule = ".triton.module";

    public const string TritonSource = ".triton.source";

    public const string CudaFunctionMeta = ".cuda.func";
}
