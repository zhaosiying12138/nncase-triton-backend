// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CodeGen.NTT.CUDA;

internal sealed class CudaLinkedModule : ILinkedModule
{
    public CudaLinkedModule(IReadOnlyList<ILinkedFunction> functions, IReadOnlyList<ILinkedSection> sections)
    {
        Functions = functions;
        Sections = sections;
    }

    public string ModuleKind => "cuda";

    public uint Version => 0;

    public IReadOnlyList<ILinkedFunction> Functions { get; }

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
