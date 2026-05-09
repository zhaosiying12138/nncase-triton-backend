// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.CodeGen.NTT.CUDA;

internal sealed record CudaFunctionMemoryPoolDesc(
    ulong RdataPoolSize,
    ulong ThreadLocalRdataPoolSize,
    ulong BlockLocalRdataPoolSize);

internal sealed class CudaLinkableFunction : ILinkableFunction
{
    public CudaLinkableFunction(
        uint id,
        BaseFunction sourceFunction,
        CudaTritonFunctionSource functionSource,
        CudaFunctionMemoryPoolDesc memoryPoolDesc,
        Stream text,
        params ILinkedSection[] sections)
    {
        Id = id;
        SourceFunction = sourceFunction;
        FunctionSource = functionSource;
        MemoryPoolDesc = memoryPoolDesc;
        Text = text;
        Sections = sections;
    }

    public uint Id { get; }

    public BaseFunction SourceFunction { get; }

    public CudaTritonFunctionSource FunctionSource { get; }

    public CudaFunctionMemoryPoolDesc MemoryPoolDesc { get; }

    public Stream Text { get; }

    public IEnumerable<FunctionRef> FunctionRefs => Enumerable.Empty<FunctionRef>();

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
