// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Reactive;
using System.Text;
using Nncase;
using Nncase.IR;
using Nncase.TIR;
using NttTir = Nncase.TIR.NTT;

namespace Nncase.CodeGen.NTT.CUDA;

internal sealed class CudaTritonLaunchCollector : ExprWalker
{
    private readonly List<CudaTritonKernelLaunch> _launches = new();
    private readonly List<CudaTritonReturnDesc> _returns = new();
    private readonly Dictionary<IVar, TIR.Buffer> _bufferViewBindings = new(ReferenceEqualityComparer.Instance);

    public IReadOnlyList<CudaTritonReturnDesc> Returns => _returns;

    public IReadOnlyList<CudaTritonKernelLaunch> Collect(PrimFunction function)
    {
        Visit(function.Body);
        return _launches;
    }

    protected override Unit VisitCall(Call expr)
    {
        if (TryGetFunctionCallName(expr.Target, out var functionName))
        {
            AddFunctionLaunch(expr, functionName);
            return default;
        }

        if (expr.Target is NttTir.NTTKernelOp or Memcopy)
        {
            AddLaunch(expr);
            return default;
        }

        return base.VisitCall(expr);
    }

    protected override Unit VisitLet(Let expr)
    {
        var var = expr.Var;
        var hasOldBinding = _bufferViewBindings.TryGetValue(var, out var oldBinding);
        var hasNewBinding = TryGetBufferDescriptor(expr.Expression, out var buffer);
        if (hasNewBinding)
        {
            _bufferViewBindings[var] = buffer;
        }

        Visit(expr.Expression);
        Visit(expr.Body);

        if (hasNewBinding)
        {
            if (hasOldBinding)
            {
                _bufferViewBindings[var] = oldBinding!;
            }
            else
            {
                _bufferViewBindings.Remove(var);
            }
        }

        return default;
    }

    protected override Unit VisitReturn(Return expr)
    {
        foreach (var value in FlattenTuple(expr.Values.ToArray()))
        {
            if (value is TIR.Buffer buffer)
            {
                _returns.Add(new CudaTritonReturnDesc(
                    _returns.Count,
                    buffer.Name,
                    DescribeBuffer(buffer)));
            }
        }

        return default;
    }

    private CudaTritonLaunchKind GetLaunchKind(BaseExpr target) => target switch
    {
        NttTir.TensorLoad or NttTir.TensorStore => CudaTritonLaunchKind.Boxing,
        NttTir.GatherReduceScatter or NttTir.SynchronizeThreads => CudaTritonLaunchKind.Collective,
        NttTir.PagedAttention => CudaTritonLaunchKind.Collective,
        NttTir.Reduce or NttTir.ReduceArg => CudaTritonLaunchKind.Reduce,
        NttTir.Matmul or NttTir.PackedMatMul or NttTir.SUMMA => CudaTritonLaunchKind.Matmul,
        Memcopy => CudaTritonLaunchKind.Memcopy,
        _ => CudaTritonLaunchKind.Compute,
    };

    private bool RequiresSeparateLaunch(CudaTritonLaunchKind kind) => kind switch
    {
        CudaTritonLaunchKind.Boxing or CudaTritonLaunchKind.Reduce or CudaTritonLaunchKind.Collective => true,
        _ => false,
    };

    private string GetOpName(BaseExpr target) => target switch
    {
        NttTir.Unary unary => $"elementwise.{ToSnakeCase(unary.UnaryOp.ToString())}",
        NttTir.VectorizedBinary binary => $"elementwise.{ToSnakeCase(binary.BinaryOp.ToString())}",
        NttTir.Clamp => "elementwise.clamp",
        NttTir.Cast => "elementwise.cast",
        NttTir.Compare compare => $"elementwise.compare.{ToSnakeCase(compare.CompareOp.ToString())}",
        NttTir.Where => "elementwise.where",
        NttTir.Transpose => "transpose",
        NttTir.Gather => "gather",
        NttTir.Swish => "swish",
        NttTir.TensorLoad => "tensor_load",
        NttTir.TensorStore => "tensor_store",
        NttTir.GatherReduceScatter => "gather_reduce_scatter",
        NttTir.Reduce reduce => $"reduce.{ToSnakeCase(reduce.ReduceOp.ToString())}",
        NttTir.ReduceArg reduceArg => $"reduce_arg.{ToSnakeCase(reduceArg.ReduceArgOp.ToString())}",
        NttTir.Matmul => "matmul",
        NttTir.PackedMatMul => "packed_matmul",
        NttTir.SUMMA => "summa",
        Memcopy => "memcopy",
        NttTir.SynchronizeThreads => "synchronize_threads",
        PrimFunction primFunction => primFunction.Name,
        PrimFunctionWrapper primFunctionWrapper => primFunctionWrapper.Target.Name,
        FunctionWrapper { Target: PrimFunctionWrapper primFunctionWrapper } => primFunctionWrapper.Target.Name,
        FunctionWrapper { Target: BaseFunction baseFunction } => baseFunction.Name,
        BaseFunction baseFunction => baseFunction.Name,
        Op op => ToSnakeCase(op.GetType().Name),
        _ => ToSnakeCase(target.GetType().Name),
    };

    private string GetArgumentName(BaseExpr argument) => argument switch
    {
        _ when TryGetBufferViewName(argument, out var name) => name,
        None => "None",
        TensorConst { Value.Shape.IsScalar: true } scalar => Convert.ToString(scalar.Value[Array.Empty<long>()], CultureInfo.InvariantCulture) ?? string.Empty,
        TensorConst => "const",
        _ => ToSnakeCase(argument.GetType().Name),
    };

    private CudaTritonBufferDesc DescribeBuffer(TIR.Buffer buffer)
    {
        var memory = new CudaTritonMemoryDesc(
            buffer.MemSpan.Buffer.Location.ToString(),
            buffer.MemSpan.Buffer.Hierarchy,
            buffer.MemSpan.Buffer.Alignment,
            ToDimDesc(buffer.MemSpan.Buffer.Size),
            ToDimDesc(buffer.MemSpan.Start),
            ToDimDesc(buffer.MemSpan.Size),
            GetBaseStart(buffer.MemSpan.Buffer.Start));

        return new CudaTritonBufferDesc(
            buffer.Name,
            buffer.ElemType.GetCSharpName(),
            buffer.ElemType.SizeInBytes,
            buffer.Rank,
            buffer.Dimensions.ToArray().Select(ToDimDesc).ToArray(),
            buffer.Strides.ToArray().Select(ToStrideDesc).ToArray(),
            memory,
            buffer.DistributedType?.ToString());
    }

    private CudaTritonDimDesc ToDimDesc(Dimension dimension) => dimension switch
    {
        DimConst dim => new("fixed", dim.Value),
        DimVar dim => new("dynamic", Symbol: dim.Name),
        _ when dimension.IsFixed => new("fixed", dimension.FixedValue),
        _ => new(GetDimensionKind(dimension.Kind), Expression: dimension.ToString()),
    };

    private CudaTritonStrideDesc ToStrideDesc(Dimension dimension)
    {
        var desc = ToDimDesc(dimension);
        return new(desc.Kind, desc.Value, desc.Symbol, desc.Expression);
    }

    private string GetDimensionKind(DimensionKind kind) => kind switch
    {
        DimensionKind.Fixed => "fixed",
        DimensionKind.Dynamic => "dynamic",
        DimensionKind.Unknown => "unknown",
        _ => kind.ToString(),
    };

    private string? GetBaseStart(Expr start) => start switch
    {
        None => null,
        TensorConst { Value.Shape.IsScalar: true } scalar => Convert.ToString(scalar.Value[Array.Empty<long>()], CultureInfo.InvariantCulture),
        _ => start.ToString(),
    };

    private IReadOnlyDictionary<string, CudaTritonBufferDesc> GetBufferArguments(Call call)
    {
        var buffers = new Dictionary<string, CudaTritonBufferDesc>(StringComparer.Ordinal);
        var arguments = call.Arguments.ToArray();
        for (int i = 0; i < arguments.Length; i++)
        {
            if (TryGetBufferDescriptor(arguments[i], out var buffer))
            {
                var argumentName = GetArgumentName(arguments[i]);
                var desc = DescribeBuffer(buffer);
                if (!string.Equals(desc.Name, argumentName, StringComparison.Ordinal))
                {
                    desc = desc with { Name = argumentName };
                }

                buffers[GetArgumentKey(call.Target, i)] = desc;
                buffers[$"arg{i.ToString(CultureInfo.InvariantCulture)}"] = desc;
            }
        }

        return buffers;
    }

    private bool TryGetBufferDescriptor(BaseExpr argument, out TIR.Buffer buffer)
    {
        if (argument is TIR.Buffer directBuffer)
        {
            buffer = directBuffer;
            return true;
        }

        if (argument is IVar var && _bufferViewBindings.TryGetValue(var, out var boundBuffer))
        {
            buffer = boundBuffer;
            return true;
        }

        if (argument is Call call && TryGetViewBuffer(call, out var viewBuffer))
        {
            buffer = viewBuffer;
            return true;
        }

        buffer = null!;
        return false;
    }

    private bool TryGetViewBuffer(Call call, out TIR.Buffer buffer)
    {
        var arguments = call.Arguments.ToArray();
        if (call.Target is IR.Buffers.AllocateBufferView && arguments.Length == 1 && arguments[0] is TIR.Buffer viewBuffer)
        {
            buffer = viewBuffer;
            return true;
        }

        if (call.Target is IR.Buffers.BufferSubview && arguments.Length >= 1 && TryGetBufferDescriptor(arguments[0], out var subviewBuffer))
        {
            buffer = subviewBuffer;
            return true;
        }

        buffer = null!;
        return false;
    }

    private bool TryGetBufferViewName(BaseExpr expr, out string name)
    {
        switch (expr)
        {
            case TIR.Buffer buffer:
                name = buffer.Name;
                return true;
            case Var var:
                name = var.Name;
                return true;
            case Call { Target: IR.Buffers.AllocateBufferView } allocateBufferView:
                {
                    var arguments = allocateBufferView.Arguments.ToArray();
                    if (arguments.Length == 1)
                    {
                        return TryGetBufferViewName(arguments[0], out name);
                    }

                    break;
                }

            case Call { Target: IR.Buffers.BufferSubview } bufferSubview:
                {
                    var arguments = bufferSubview.Arguments.ToArray();
                    if (arguments.Length >= 1)
                    {
                        return TryGetBufferViewName(arguments[0], out name);
                    }

                    break;
                }
        }

        name = string.Empty;
        return false;
    }

    private string GetArgumentKey(BaseExpr target, int index)
    {
        if (target is Op op)
        {
            var parameter = op.Parameters.FirstOrDefault(p => p.Index == index);
            if (parameter != null)
            {
                return ToSnakeCase(parameter.Name);
            }
        }

        return $"arg{index.ToString(CultureInfo.InvariantCulture)}";
    }

    private bool TryGetFunctionCallName(BaseExpr target, out string functionName)
    {
        functionName = target switch
        {
            PrimFunction primFunction => primFunction.Name,
            Fusion fusion => fusion.Name,
            PrimFunctionWrapper primFunctionWrapper => primFunctionWrapper.Target.Name,
            FunctionWrapper { Target: PrimFunctionWrapper primFunctionWrapper } => primFunctionWrapper.Target.Name,
            FunctionWrapper { Target: BaseFunction baseFunction } => baseFunction.Name,
            _ => string.Empty,
        };
        return !string.IsNullOrEmpty(functionName);
    }

    private IEnumerable<BaseExpr> FlattenTuple(IEnumerable<BaseExpr> values)
    {
        foreach (var value in values)
        {
            if (value is IR.Tuple tuple)
            {
                foreach (var item in FlattenTuple(tuple.Fields.ToArray()))
                {
                    yield return item;
                }
            }
            else
            {
                yield return value;
            }
        }
    }

    private IReadOnlyDictionary<string, object?> GetOpAttributes(BaseExpr target) => target switch
    {
        NttTir.PagedAttention pagedAttention => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["layer_id"] = pagedAttention.LayerId,
            ["layout"] = ToStringArray(pagedAttention.Layout),
            ["hidden_size"] = pagedAttention.HiddenSize,
        },
        NttTir.UpdatePagedAttentionKVCache updateKv => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["cache_kind"] = updateKv.CacheKind.ToString(),
            ["layer_id"] = updateKv.LayerId,
            ["layout"] = ToStringArray(updateKv.Layout),
        },
        NttTir.Reduce reduce => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["reduce_op"] = reduce.ReduceOp.ToString(),
            ["axes"] = ToIntArray(reduce.Axes),
            ["keep_dims"] = reduce.KeepDims,
            ["vectorized_axes"] = ToIntArray(reduce.VectorizedAxes),
            ["padded_nums"] = ToDimDescArray(reduce.PadedNums),
        },
        NttTir.ReduceArg reduceArg => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["reduce_arg_op"] = reduceArg.ReduceArgOp.ToString(),
            ["axis"] = reduceArg.Axis,
            ["keep_dims"] = reduceArg.KeepDims,
            ["select_last_index"] = reduceArg.SelectLastIndex,
            ["dest_type"] = reduceArg.DestType.GetCSharpName(),
        },
        NttTir.Matmul matmul => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["lhs_vectorized_axes"] = ToIntArray(matmul.LhsVectorizedAxes),
            ["rhs_vectorized_axes"] = ToIntArray(matmul.RhsVectorizedAxes),
            ["transpose_a"] = matmul.TransposeA,
            ["transpose_b"] = matmul.TransposeB,
            ["fused_reduce"] = matmul.FusedReduce,
            ["c_source_path"] = matmul.CSourcePath ?? string.Empty,
            ["func_name"] = matmul.FuncName ?? string.Empty,
        },
        NttTir.PackedMatMul packedMatMul => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["fused_reduce"] = packedMatMul.FusedReduce,
        },
        NttTir.Pack pack => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["lanes"] = ToIntArray(pack.Lanes),
            ["axes"] = ToIntArray(pack.Axes),
        },
        NttTir.Unpack unpack => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["lanes"] = ToIntArray(unpack.Lanes),
            ["axes"] = ToIntArray(unpack.Axes),
        },
        NttTir.SUMMA summa => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["lhs_vectorized_axes"] = ToIntArray(summa.LhsVectorizedAxes),
            ["rhs_vectorized_axes"] = ToIntArray(summa.RhsVectorizedAxes),
            ["transpose_a"] = summa.TransposeA,
            ["transpose_b"] = summa.TransposeB,
        },
        NttTir.Unary unary => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["unary_op"] = unary.UnaryOp.ToString(),
        },
        NttTir.VectorizedBinary binary => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["binary_op"] = binary.BinaryOp.ToString(),
            ["lhs_vectorized_axes"] = ToIntArray(binary.LhsVectorizedAxes),
            ["lhs_padded_nums"] = ToDimDescArray(binary.LhsPadedNums),
            ["rhs_vectorized_axes"] = ToIntArray(binary.RhsVectorizedAxes),
            ["rhs_padded_nums"] = ToDimDescArray(binary.RhsPadedNums),
        },
        NttTir.Clamp clamp => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["min"] = clamp.Min,
            ["max"] = clamp.Max,
        },
        NttTir.Cast cast => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["new_type"] = cast.NewType.GetCSharpName(),
            ["cast_mode"] = cast.CastMode.ToString(),
            ["vectorize_axes"] = ToIntArray(cast.VectorizeAxes),
        },
        NttTir.Compare compare => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["compare_op"] = compare.CompareOp.ToString(),
        },
        NttTir.Transpose transpose => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["perm"] = ToIntArray(transpose.Perm),
        },
        NttTir.Gather gather => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["axis"] = gather.Axis,
        },
        NttTir.Concat concat => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["axis"] = concat.Axis,
        },
        NttTir.Swish swish => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["beta"] = swish.Beta,
        },
        NttTir.VectorizedLayerNorm layerNorm => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["axis"] = layerNorm.Axis,
            ["epsilon"] = layerNorm.Epsilon,
            ["use_mean"] = layerNorm.UseMean,
            ["vectorized_axes"] = ToIntArray(layerNorm.VectorizedAxes),
            ["padded_nums"] = ToDimDescArray(layerNorm.PadedNums),
            ["c_source_path"] = layerNorm.CSourcePath ?? string.Empty,
            ["func_name"] = layerNorm.FuncName ?? string.Empty,
        },
        NttTir.TensorLoad tensorLoad => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["nd_sbp"] = ToStringArray(tensorLoad.NdSbp),
            ["placement"] = tensorLoad.Placement.ToString(),
        },
        NttTir.TensorStore tensorStore => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["nd_sbp"] = ToStringArray(tensorStore.NdSbp),
            ["placement"] = tensorStore.Placement.ToString(),
        },
        NttTir.GatherReduceScatter grs => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["in_type"] = grs.InType.ToString(),
            ["out_type"] = grs.OutType.ToString(),
        },
        NttTir.SynchronizeThreads => new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["scope"] = "threads",
        },
        _ => new Dictionary<string, object?>(StringComparer.Ordinal),
    };

    private int[] ToIntArray(IRArray<int> values)
        => values.IsDefaultOrEmpty ? Array.Empty<int>() : values.ToArray();

    private CudaTritonDimDesc[] ToDimDescArray(IRArray<Dimension> values)
        => values.IsDefaultOrEmpty ? Array.Empty<CudaTritonDimDesc>() : values.Select(ToDimDesc).ToArray();

    private string[] ToStringArray<T>(IRArray<T> values)
        => values.IsDefaultOrEmpty ? Array.Empty<string>() : values.Select(value => value?.ToString() ?? string.Empty).ToArray();

    private string ToSnakeCase(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return string.Empty;
        }

        var builder = new StringBuilder(value.Length + 8);
        for (int i = 0; i < value.Length; i++)
        {
            var ch = value[i];
            if (char.IsUpper(ch))
            {
                if (i != 0 && builder[^1] != '_' && !char.IsUpper(value[i - 1]))
                {
                    builder.Append('_');
                }

                builder.Append(char.ToLowerInvariant(ch));
            }
            else if (char.IsLetterOrDigit(ch))
            {
                builder.Append(ch);
            }
            else if (builder.Length > 0 && builder[^1] != '_')
            {
                builder.Append('_');
            }
        }

        return builder.ToString().Trim('_');
    }

    private void AddLaunch(Call call)
    {
        var kind = GetLaunchKind(call.Target);
        var arguments = call.Arguments.ToArray().Select(GetArgumentName).ToArray();
        var bufferArguments = GetBufferArguments(call);
        var opAttributes = GetOpAttributes(call.Target);
        var ordinal = _launches.Count;
        _launches.Add(new CudaTritonKernelLaunch(
            ordinal,
            kind,
            GetOpName(call.Target),
            arguments,
            RequiresSeparateLaunch(kind),
            bufferArguments,
            opAttributes));

        if (call.Target is NttTir.Matmul { FusedReduce: true } or NttTir.PackedMatMul { FusedReduce: true })
        {
            _launches.Add(new CudaTritonKernelLaunch(
                _launches.Count,
                CudaTritonLaunchKind.Reduce,
                "matmul.fused_reduce",
                arguments,
                true,
                bufferArguments,
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["source_op"] = GetOpName(call.Target),
                    ["fused_reduce"] = true,
                }));
        }
    }

    private void AddFunctionLaunch(Call call, string functionName)
    {
        var arguments = call.Arguments.ToArray().Select(GetArgumentName).ToArray();
        _launches.Add(new CudaTritonKernelLaunch(
            _launches.Count,
            CudaTritonLaunchKind.Function,
            functionName,
            arguments,
            false,
            GetBufferArguments(call),
            new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["function_name"] = functionName,
            }));
    }
}
