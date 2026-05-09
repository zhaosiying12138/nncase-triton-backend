// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Targets;

namespace Nncase.CodeGen.NTT.CUDA;

public sealed class CudaModuleBuilder : IModuleBuilder
{
    private readonly SectionManager _sectionManager;
    private readonly BinaryWriter _rdataWriter;
    private readonly BinaryWriter[] _threadLocalRdataWriters;
    private readonly BinaryWriter[] _blockLocalRdataWriters;
    private readonly NTTTargetOptions _targetOptions;

    public CudaModuleBuilder(CompileOptions options)
    {
        CompileOptions = options;
        _targetOptions = options.TargetOptions as NTTTargetOptions ?? new NTTTargetOptions();
        _sectionManager = new();
        _rdataWriter = _sectionManager.GetWriter(WellknownSectionNames.Rdata);

        var shardCount = TensorUtilities.GetProduct(_targetOptions.Hierarchies[0]);
        var blockCount = Math.Max(1, shardCount / _targetOptions.Hierarchies[0][^1]);
        _threadLocalRdataWriters = new BinaryWriter[shardCount];
        _blockLocalRdataWriters = new BinaryWriter[blockCount];
        for (int i = 0; i < shardCount; i++)
        {
            _threadLocalRdataWriters[i] = _sectionManager.GetWriter(WellknownSectionNames.ThreadLocalRdata, i);
        }

        for (int i = 0; i < blockCount; i++)
        {
            _blockLocalRdataWriters[i] = _sectionManager.GetWriter(WellknownSectionNames.BlockLocalRdata, i);
        }
    }

    public CompileOptions CompileOptions { get; }

    public string ModuleKind => "cuda";

    public ILinkableModule Build(IReadOnlyList<BaseFunction> functions)
    {
        var linkableFunctions = functions
            .Select((f, i) => new CudaFunctionBuilder(
                (uint)i,
                _rdataWriter,
                _threadLocalRdataWriters,
                _blockLocalRdataWriters,
                _targetOptions).Build(f))
            .Cast<CudaLinkableFunction>()
            .ToArray();

        _rdataWriter.Flush();
        var threadLocalRdataContents = Enumerable.Range(0, _threadLocalRdataWriters.Length).Select(i =>
        {
            _threadLocalRdataWriters[i].Flush();
            return _sectionManager.GetContent(WellknownSectionNames.ThreadLocalRdata, i)!;
        }).ToArray();
        var blockLocalRdataContents = Enumerable.Range(0, _blockLocalRdataWriters.Length).Select(i =>
        {
            _blockLocalRdataWriters[i].Flush();
            return _sectionManager.GetContent(WellknownSectionNames.BlockLocalRdata, i)!;
        }).ToArray();

        return new CudaLinkableModule(
            _sectionManager.GetContent(WellknownSectionNames.Rdata)!,
            threadLocalRdataContents,
            blockLocalRdataContents,
            linkableFunctions,
            _targetOptions);
    }
}
