// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text;
using Nncase.IR;
using Nncase.Targets;
using Nncase.Utilities;

namespace Nncase.CodeGen.NTT.CUDA;

internal sealed class CudaFunctionBuilder
{
    private static readonly HashSet<string> ExcludedKernelParameters = new(StringComparer.Ordinal)
    {
        "data",
        "block_local_data",
    };

    private readonly uint _id;
    private readonly BinaryWriter _rdataWriter;
    private readonly IReadOnlyList<BinaryWriter> _threadLocalRdataWriters;
    private readonly IReadOnlyList<BinaryWriter> _blockLocalRdataWriters;
    private readonly NTTTargetOptions _targetOptions;

    public CudaFunctionBuilder(
        uint id,
        BinaryWriter rdataWriter,
        IReadOnlyList<BinaryWriter> threadLocalRdataWriters,
        IReadOnlyList<BinaryWriter> blockLocalRdataWriters,
        NTTTargetOptions targetOptions)
    {
        _id = id;
        _rdataWriter = rdataWriter;
        _threadLocalRdataWriters = threadLocalRdataWriters;
        _blockLocalRdataWriters = blockLocalRdataWriters;
        _targetOptions = targetOptions;
    }

    public ILinkableFunction Build(BaseFunction baseFunc) => baseFunc switch
    {
        TIR.PrimFunction primFunc => BuildPrimFunction(primFunc),
        Fusion fusion => BuildFusion(fusion),
        _ => throw new NotSupportedException($"the {baseFunc.GetType()} {baseFunc.Name} is not supported for cuda triton codegen!"),
    };

    private CudaLinkableFunction BuildPrimFunction(TIR.PrimFunction primFunc)
    {
        var rdataPoolSize = WriteRdata(primFunc);
        var threadLocalRdataPoolSize = WriteThreadLocalRdata(primFunc);
        var blockLocalRdataPoolSize = WriteBlockLocalRdata(primFunc);
        var memoryPoolDesc = new CudaFunctionMemoryPoolDesc(
            rdataPoolSize,
            threadLocalRdataPoolSize,
            blockLocalRdataPoolSize);
        var collector = new CudaTritonLaunchCollector();
        var launches = collector.Collect(primFunc);
        var returns = collector.Returns;
        var parameters = GetParameterNames(primFunc.Parameters);
        var functionSource = new CudaTritonFunctionSource(
            _id,
            primFunc.Name,
            primFunc.IsEntry,
            parameters,
            primFunc.SchedResult.DataUsage,
            primFunc.SchedResult.OutputUsage,
            rdataPoolSize,
            launches,
            primFunc.SchedResult.BlockLocalDataPoolSize,
            GetFunctionBuffers(launches, returns),
            GetParameterDescs(primFunc.Parameters),
            returns);

        return CreateLinkableFunction(primFunc, functionSource, memoryPoolDesc);
    }

    private CudaLinkableFunction BuildFusion(Fusion fusion)
    {
        var launch = new CudaTritonKernelLaunch(
            0,
            CudaTritonLaunchKind.Compute,
            $"fusion.{fusion.Name}",
            GetParameterNames(fusion.Parameters),
            false);
        var functionSource = new CudaTritonFunctionSource(
            _id,
            fusion.Name,
            fusion.IsEntry,
            GetParameterNames(fusion.Parameters),
            fusion.SchedResult.DataUsage,
            fusion.SchedResult.OutputUsage,
            0,
            [launch],
            fusion.SchedResult.BlockLocalDataPoolSize,
            new Dictionary<string, CudaTritonBufferDesc>(StringComparer.Ordinal),
            GetParameterDescs(fusion.Parameters),
            Array.Empty<CudaTritonReturnDesc>());

        return CreateLinkableFunction(fusion, functionSource, new(0, 0, 0));
    }

    private CudaLinkableFunction CreateLinkableFunction(
        BaseFunction sourceFunction,
        CudaTritonFunctionSource functionSource,
        CudaFunctionMemoryPoolDesc memoryPoolDesc)
    {
        var text = new MemoryStream(Encoding.UTF8.GetBytes($"cuda.triton.func:{functionSource.Id}:{functionSource.Name}\n"));
        var functionMeta = new MemoryStream(Encoding.UTF8.GetBytes(new TritonPythonSourceBuilder().BuildMetadataJson(new CudaTritonModuleSource(
            PeCount: TensorUtilities.GetProduct(_targetOptions.Hierarchies[0]),
            RdataPoolSize: memoryPoolDesc.RdataPoolSize,
            ThreadLocalRdataPoolSize: memoryPoolDesc.ThreadLocalRdataPoolSize,
            BlockLocalRdataPoolSize: memoryPoolDesc.BlockLocalRdataPoolSize,
            Functions: [functionSource],
            FusedKernelMode: _targetOptions.FusedKernelMode))));
        var functionMetaSection = new LinkedSection(
            functionMeta,
            CudaSectionNames.CudaFunctionMeta,
            0,
            8,
            (ulong)functionMeta.Length);

        return new CudaLinkableFunction(
            _id,
            sourceFunction,
            functionSource,
            memoryPoolDesc,
            text,
            functionMetaSection);
    }

    private ulong WriteRdata(TIR.PrimFunction primFunc)
    {
        ulong rdataPoolSize = 0;
        foreach (var (@const, range) in primFunc.SchedResult.Rdatas)
        {
            var tensor = ((TensorConst)@const).Value;
            var size = range.Max - range.Min;
            rdataPoolSize = Math.Max(range.Max, rdataPoolSize);
            if ((ulong)tensor.Length * (ulong)tensor.ElementType.SizeInBytes != size)
            {
                throw new InvalidDataException("The Buffer Size Not Equal!");
            }

            _rdataWriter.Position(checked((long)range.Min));
            tensor.Serialize(_rdataWriter.BaseStream);
        }

        return rdataPoolSize;
    }

    private ulong WriteThreadLocalRdata(TIR.PrimFunction primFunc)
    {
        ulong threadLocalRdataPoolSize = 0;
        foreach (var (@const, range) in primFunc.SchedResult.ThreadLocalRdatas)
        {
            var tensor = ((TensorConst)@const).Value;
            var distributedType = (DistributedType)@const.CheckedType;
            var size = range.Max - range.Min;
            threadLocalRdataPoolSize = Math.Max(range.Max, threadLocalRdataPoolSize);
            var dividedDims = DistributedUtility.GetDividedTensorType(distributedType).Shape.ToValueArray();
            var localStrides = TensorUtilities.GetDefaultStrides(dividedDims);
            for (int i = 0; i < _threadLocalRdataWriters.Count; i++)
            {
                var shardIndex = DistributedUtility.GetUnraveledIndex(i, _targetOptions.Hierarchies[0]);
                (var localOffset, var localShape) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
                var linearOffset = TensorUtilities.GetLinearOffset(tensor.Strides, localOffset);

                if ((ulong)TensorUtilities.GetProduct(localShape) * (ulong)tensor.ElementType.SizeInBytes > size)
                {
                    throw new InvalidDataException("The Buffer Size Not Equal!");
                }

                var writer = _threadLocalRdataWriters[i];
                writer.Position(checked((long)range.Min));
                tensor.Serialize(writer.BaseStream, linearOffset, localShape, localStrides);
            }
        }

        return threadLocalRdataPoolSize;
    }

    private ulong WriteBlockLocalRdata(TIR.PrimFunction primFunc)
    {
        ulong blockLocalRdataPoolSize = 0;
        foreach (var (@const, range) in primFunc.SchedResult.BlockLocalRdatas)
        {
            var tensor = ((TensorConst)@const).Value;
            var distributedType = (DistributedType)@const.CheckedType;
            var size = range.Max - range.Min;
            blockLocalRdataPoolSize = Math.Max(range.Max, blockLocalRdataPoolSize);
            var dividedDims = DistributedUtility.GetDividedTensorType(distributedType).Shape.ToValueArray();
            var localStrides = TensorUtilities.GetDefaultStrides(dividedDims);
            var blockHierarchy = _targetOptions.Hierarchies[0][..^1];
            for (int i = 0; i < _blockLocalRdataWriters.Count; i++)
            {
                var shardIndex = DistributedUtility.GetUnraveledIndex(i, blockHierarchy).Concat([0]).ToArray();
                (var localOffset, var localShape) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
                var linearOffset = TensorUtilities.GetLinearOffset(tensor.Strides, localOffset);

                if ((ulong)TensorUtilities.GetProduct(localShape) * (ulong)tensor.ElementType.SizeInBytes > size)
                {
                    throw new InvalidDataException("The Buffer Size Not Equal!");
                }

                var writer = _blockLocalRdataWriters[i];
                writer.Position(checked((long)range.Min));
                tensor.Serialize(writer.BaseStream, linearOffset, localShape, localStrides);
            }
        }

        return blockLocalRdataPoolSize;
    }

    private IReadOnlyList<string> GetParameterNames(ReadOnlySpan<IVar> parameters)
        => parameters.ToArray()
            .OfType<Var>()
            .Where(v => !ExcludedKernelParameters.Contains(v.Name))
            .Select(v => v.Name)
            .ToArray();

    private IReadOnlyList<CudaTritonFunctionParameterDesc> GetParameterDescs(ReadOnlySpan<IVar> parameters)
        => parameters.ToArray()
            .OfType<Var>()
            .Where(v => !ExcludedKernelParameters.Contains(v.Name))
            .Select(v => new CudaTritonFunctionParameterDesc(v.Name, TryGetParameterType(v)))
            .ToArray();

    private string? TryGetParameterType(Var parameter)
    {
        try
        {
            return parameter.CheckedType.ToString();
        }
        catch (Exception)
        {
            return null;
        }
    }

    private IReadOnlyDictionary<string, CudaTritonBufferDesc> GetFunctionBuffers(
        IReadOnlyList<CudaTritonKernelLaunch> launches,
        IReadOnlyList<CudaTritonReturnDesc> returns)
    {
        var buffers = new Dictionary<string, CudaTritonBufferDesc>(StringComparer.Ordinal);
        foreach (var launch in launches)
        {
            if (launch.BufferArguments == null)
            {
                continue;
            }

            foreach (var buffer in launch.BufferArguments.Values)
            {
                buffers.TryAdd(buffer.Name, buffer);
            }
        }

        foreach (var ret in returns)
        {
            buffers.TryAdd(ret.Buffer.Name, ret.Buffer);
        }

        return buffers;
    }
}
