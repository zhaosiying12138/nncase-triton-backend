// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.RegularExpressions;
using Nncase.Targets;

namespace Nncase.CodeGen.NTT.CUDA;

internal static class CudaComputeCclTailPlanner
{
    private const string CclTailAttrsKey = "ccl_tails";

    public static CudaTritonModuleSource Plan(CudaTritonModuleSource module)
    {
        if (module.FusedKernelMode != CudaFusedKernelMode.ComputeCcl)
        {
            return module;
        }

        if (module.Functions.SelectMany(f => f.Launches).Any(l => l.OpAttributes?.ContainsKey("elided_by_ccl_tail") == true))
        {
            return module;
        }

        var functionMap = module.Functions.ToDictionary(f => f.Name, StringComparer.Ordinal);
        var functions = module.Functions.ToDictionary(f => f.Name, f => f, StringComparer.Ordinal);
        var functionCallCounts = module.Functions
            .SelectMany(f => f.Launches)
            .Where(l => l.Kind == CudaTritonLaunchKind.Function)
            .GroupBy(l => l.OpName, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.Count(), StringComparer.Ordinal);
        var fusedSites = 0;

        foreach (var function in module.Functions)
        {
            var launches = function.Launches.ToArray();
            var producers = BuildProducerMap(function, functionMap, functionCallCounts);
            var producerConsumerCounts = CountSingleProducerGrsConsumers(launches, producers);
            var changed = false;

            for (int i = 0; i < launches.Length; i++)
            {
                var grs = launches[i];
                if (!IsGatherReduceScatter(grs) || !TryGetArgumentBuffer(grs, 0, out var srcDesc) || !TryGetArgumentBuffer(grs, 1, out var dstDesc))
                {
                    continue;
                }

                var srcKey = BufferKey.From(srcDesc);
                if (!producers.TryGetValue(srcKey, out var matches) || matches.Count != 1)
                {
                    launches[i] = WithAttrs(grs, attrs =>
                    {
                        attrs["ccl_tail_fusion"] = "unfused";
                        attrs["ccl_tail_unfused_reason"] = matches is null ? "no_unique_producer" : $"producer_count={matches.Count.ToString(CultureInfo.InvariantCulture)}";
                    });
                    changed = true;
                    continue;
                }

                var producer = matches[0];
                var producerKey = ProducerSiteKey.From(producer);
                if (producerConsumerCounts.TryGetValue(producerKey, out var consumerCount) && consumerCount != 1)
                {
                    launches[i] = WithAttrs(grs, attrs =>
                    {
                        attrs["ccl_tail_fusion"] = "unfused";
                        attrs["ccl_tail_unfused_reason"] = $"producer_tail_count={consumerCount.ToString(CultureInfo.InvariantCulture)}";
                    });
                    changed = true;
                    continue;
                }

                var producerFunction = functions[producer.Function.Name];
                var producerLaunches = producerFunction.Launches.ToArray();
                var producerLaunch = producerLaunches[producer.LaunchIndex];
                var tail = CreateTailAttrs(function, grs, producer, producerLaunch, srcDesc, dstDesc, fusedSites);
                var updatedProducerLaunch = AddTail(producerLaunch, tail);
                producerLaunches[producer.LaunchIndex] = updatedProducerLaunch;
                functions[producer.Function.Name] = producerFunction with { Launches = producerLaunches };

                if (string.Equals(producer.Function.Name, function.Name, StringComparison.Ordinal))
                {
                    launches[producer.LaunchIndex] = updatedProducerLaunch;
                }

                if (producer.ParentFunction is { } parentFunction &&
                    producer.ParentFunctionLaunchIndex.HasValue &&
                    string.Equals(parentFunction.Name, function.Name, StringComparison.Ordinal))
                {
                    var parentIndex = producer.ParentFunctionLaunchIndex.Value;
                    var parentLaunch = launches[parentIndex];
                    launches[parentIndex] = WithAttrs(parentLaunch, attrs =>
                    {
                        attrs["ccl_tail_nested_writer_ordinal"] = producer.Launch.Ordinal;
                        attrs["ccl_tail_nested_writer_function"] = producer.Function.Name;
                    });
                }

                launches[i] = WithAttrs(grs, attrs =>
                {
                    attrs["elided_by_ccl_tail"] = true;
                    attrs["ccl_tail_producer_function"] = producer.Function.Name;
                    attrs["ccl_tail_producer_ordinal"] = producer.Launch.Ordinal;
                    attrs["ccl_tail_producer_op"] = producer.Launch.OpName;
                });
                fusedSites++;
                changed = true;
            }

            if (changed)
            {
                functions[function.Name] = functions[function.Name] with { Launches = launches };
            }
        }

        return module with { Functions = module.Functions.Select(f => functions[f.Name]).ToArray() };
    }

    public static ulong GetScratchBytes(CudaTritonModuleSource module)
    {
        var tailCount = module.Functions
            .SelectMany(f => f.Launches)
            .SelectMany(l => GetTailList(l.OpAttributes))
            .Count();
        return checked((ulong)tailCount * 16UL);
    }

    private static Dictionary<ProducerSiteKey, int> CountSingleProducerGrsConsumers(
        IReadOnlyList<CudaTritonKernelLaunch> launches,
        IReadOnlyDictionary<BufferKey, List<ProducerRef>> producers)
    {
        var counts = new Dictionary<ProducerSiteKey, int>();
        foreach (var grs in launches)
        {
            if (!IsGatherReduceScatter(grs) || !TryGetArgumentBuffer(grs, 0, out var srcDesc))
            {
                continue;
            }

            if (!producers.TryGetValue(BufferKey.From(srcDesc), out var matches) || matches.Count != 1)
            {
                continue;
            }

            var key = ProducerSiteKey.From(matches[0]);
            counts[key] = counts.TryGetValue(key, out var value) ? value + 1 : 1;
        }

        return counts;
    }

    private static Dictionary<BufferKey, List<ProducerRef>> BuildProducerMap(
        CudaTritonFunctionSource function,
        IReadOnlyDictionary<string, CudaTritonFunctionSource> functionMap,
        IReadOnlyDictionary<string, int> functionCallCounts)
    {
        var producers = new Dictionary<BufferKey, List<ProducerRef>>();
        var launches = function.Launches.ToArray();
        for (int launchIndex = 0; launchIndex < launches.Length; launchIndex++)
        {
            var launch = launches[launchIndex];
            if (launch.Kind == CudaTritonLaunchKind.Function && functionMap.TryGetValue(launch.OpName, out var callee))
            {
                foreach (var nested in FindNestedWriters(function, launchIndex, launch, callee, functionCallCounts))
                {
                    AddProducer(producers, nested.OutputKey, nested);
                }

                continue;
            }

            if (!IsEligibleProducer(launch))
            {
                continue;
            }

            foreach (var output in GetOutputBuffers(function, launch))
            {
                if (output.Desc is { } desc)
                {
                    AddProducer(producers, BufferKey.From(desc), new ProducerRef(function, launchIndex, launch, BufferKey.From(desc)));
                }
            }
        }

        return producers;
    }

    private static IEnumerable<ProducerRef> FindNestedWriters(
        CudaTritonFunctionSource parent,
        int parentLaunchIndex,
        CudaTritonKernelLaunch parentLaunch,
        CudaTritonFunctionSource callee,
        IReadOnlyDictionary<string, int> functionCallCounts)
    {
        if (!functionCallCounts.TryGetValue(callee.Name, out var callCount) || callCount != 1)
        {
            yield break;
        }

        var parentArgs = parentLaunch.Arguments.ToArray();
        var parameters = callee.Parameters.ToArray();
        var count = Math.Min(parentArgs.Length, parameters.Length);
        for (int parameterIndex = 0; parameterIndex < count; parameterIndex++)
        {
            if (!TryGetArgumentBuffer(parentLaunch, parameterIndex, out var parentDesc))
            {
                continue;
            }

            var parentKey = BufferKey.From(parentDesc);
            var parentMemoryKey = MemoryKey.From(parentDesc);
            var parameterName = parameters[parameterIndex];
            var parameterMemoryKey = TryGetFunctionBuffer(callee, parameterName, out var parameterDesc)
                ? MemoryKey.From(parameterDesc)
                : null;
            var nestedWriters = new List<(int Index, CudaTritonKernelLaunch Launch)>();
            var calleeLaunches = callee.Launches.ToArray();
            for (int nestedIndex = 0; nestedIndex < calleeLaunches.Length; nestedIndex++)
            {
                var nestedLaunch = calleeLaunches[nestedIndex];
                if (!IsEligibleProducer(nestedLaunch))
                {
                    continue;
                }

                foreach (var output in GetOutputBuffers(callee, nestedLaunch))
                {
                    var outputMemoryKey = output.Desc is null ? null : MemoryKey.From(output.Desc);
                    if (string.Equals(output.ArgumentName, parameterName, StringComparison.Ordinal) ||
                        string.Equals(output.Desc?.Name, parameterName, StringComparison.Ordinal) ||
                        (parameterMemoryKey is not null && outputMemoryKey is not null && outputMemoryKey.Equals(parameterMemoryKey)) ||
                        (outputMemoryKey is not null && outputMemoryKey.Equals(parentMemoryKey)))
                    {
                        nestedWriters.Add((nestedIndex, nestedLaunch));
                        break;
                    }
                }
            }

            if (nestedWriters.Count == 1)
            {
                var writer = nestedWriters[0];
                if (writer.Index != calleeLaunches.Length - 1)
                {
                    continue;
                }

                yield return new ProducerRef(
                    callee,
                    writer.Index,
                    writer.Launch,
                    parentKey,
                    parent,
                    parentLaunchIndex);
            }
        }
    }

    private static void AddProducer(Dictionary<BufferKey, List<ProducerRef>> producers, BufferKey key, ProducerRef producer)
    {
        if (!producers.TryGetValue(key, out var list))
        {
            list = new List<ProducerRef>();
            producers.Add(key, list);
        }

        list.Add(producer);
    }

    private static IReadOnlyDictionary<string, object?> CreateTailAttrs(
        CudaTritonFunctionSource sourceFunction,
        CudaTritonKernelLaunch grs,
        ProducerRef producer,
        CudaTritonKernelLaunch producerLaunch,
        CudaTritonBufferDesc srcDesc,
        CudaTritonBufferDesc dstDesc,
        int site)
    {
        var producerName = SanitizeName(producerLaunch.OpName);
        var grsAttrs = CloneAttrs(grs.OpAttributes);
        grsAttrs["ccl_tail"] = true;
        return new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["op_name"] = $"ccl_tail.{producerLaunch.OpName}.grs",
            ["kernel_name"] = $"_nncase_pe_{producerName}_grs_tail_kernel",
            ["inline_device_tail"] = true,
            ["site"] = site,
            ["source_function"] = sourceFunction.Name,
            ["source_ordinal"] = grs.Ordinal,
            ["producer_function"] = producer.Function.Name,
            ["producer_ordinal"] = producer.Launch.Ordinal,
            ["producer_op"] = producerLaunch.OpName,
            ["arguments"] = grs.Arguments.ToArray(),
            ["buffer_arguments"] = ToObjectMap(grs.BufferArguments ?? new Dictionary<string, CudaTritonBufferDesc>(StringComparer.Ordinal)
            {
                ["arg0"] = srcDesc,
                ["arg1"] = dstDesc,
            }),
            ["op_attrs"] = grsAttrs,
            ["requires_collective"] = true,
        };
    }

    private static CudaTritonKernelLaunch AddTail(CudaTritonKernelLaunch launch, IReadOnlyDictionary<string, object?> tail)
    {
        return WithAttrs(launch, attrs =>
        {
            var tails = GetTailList(attrs).Select(CloneTail).ToList();
            tails.Add(new Dictionary<string, object?>(tail, StringComparer.Ordinal));
            attrs[CclTailAttrsKey] = tails;
        });
    }

    private static IEnumerable<OutputBufferRef> GetOutputBuffers(CudaTritonFunctionSource function, CudaTritonKernelLaunch launch)
    {
        foreach (var index in GetOutputIndices(launch))
        {
            var argumentName = index < launch.Arguments.Count ? launch.Arguments[index] : string.Empty;
            yield return TryGetArgumentBuffer(launch, index, out var desc) ||
                TryGetFunctionBuffer(function, argumentName, out desc)
                    ? new OutputBufferRef(index, argumentName, desc)
                    : new OutputBufferRef(index, argumentName, null);
        }
    }

    private static IReadOnlyList<int> GetOutputIndices(CudaTritonKernelLaunch launch)
    {
        var count = launch.Arguments.Count;
        var opName = launch.OpName;
        if (opName.StartsWith("fusion.cuda.flash_attention", StringComparison.Ordinal))
        {
            return count > 3 ? [3] : [];
        }

        if (opName.StartsWith("fusion.cuda.swish_mul", StringComparison.Ordinal) ||
            opName.StartsWith("fusion.cuda.mul_cos", StringComparison.Ordinal) ||
            opName.StartsWith("fusion.cuda.mul_sin", StringComparison.Ordinal))
        {
            return count > 2 ? [2] : [];
        }

        if (opName.StartsWith("fusion.cuda.matmul_swish_mul", StringComparison.Ordinal) ||
            opName.StartsWith("fusion.cuda.silu_mul_matmul", StringComparison.Ordinal) ||
            opName.StartsWith("fusion.cuda.layer_norm_transpose", StringComparison.Ordinal) ||
            opName.StartsWith("fusion.cuda.rope", StringComparison.Ordinal))
        {
            return count > 3 ? [3] : [];
        }

        if (opName.StartsWith("fusion.cuda.matmul_mul_matmul", StringComparison.Ordinal) ||
            opName.StartsWith("fusion.cuda.matmul_silu_matmul_mul_matmul", StringComparison.Ordinal) ||
            opName.StartsWith("fusion.cuda.layer_norm_matmul", StringComparison.Ordinal))
        {
            return count > 4 ? [4] : [];
        }

        return opName switch
        {
            "memcopy" => count > 0 ? [0] : [],
            "matmul" => count > 2 ? [2] : [],
            "softmax" or "vectorized_softmax" => count > 1 ? [1] : [],
            "tensor_load" => count > 0 ? [0] : [],
            "tensor_store" => count > 1 ? [1] : [],
            "gather_reduce_scatter" => count > 1 ? [1] : [],
            "update_paged_attention_kvcache" or "update_paged_attention_kv_cache" => [],
            "paged_attention" => count > 4 ? [4] : [],
            "concat" => count > 0 ? [count - 1] : [],
            _ => count > 0 ? [count - 1] : [],
        };
    }

    private static bool TryGetArgumentBuffer(CudaTritonKernelLaunch launch, int index, out CudaTritonBufferDesc desc)
    {
        desc = null!;
        if (launch.BufferArguments is null)
        {
            return false;
        }

        if (launch.BufferArguments.TryGetValue($"arg{index.ToString(CultureInfo.InvariantCulture)}", out var byIndex))
        {
            desc = byIndex;
            return true;
        }

        if (index < launch.Arguments.Count && launch.BufferArguments.TryGetValue(launch.Arguments[index], out var byName))
        {
            desc = byName;
            return true;
        }

        return false;
    }

    private static bool TryGetFunctionBuffer(CudaTritonFunctionSource function, string name, out CudaTritonBufferDesc desc)
    {
        desc = null!;
        if (function.Buffers is null || string.IsNullOrEmpty(name))
        {
            return false;
        }

        return function.Buffers.TryGetValue(name, out desc!);
    }

    private static bool IsGatherReduceScatter(CudaTritonKernelLaunch launch)
        => launch.Kind == CudaTritonLaunchKind.Collective && string.Equals(launch.OpName, "gather_reduce_scatter", StringComparison.Ordinal);

    private static bool IsEligibleProducer(CudaTritonKernelLaunch launch)
    {
        if (launch.RequiresCollective || launch.Kind == CudaTritonLaunchKind.Collective)
        {
            return false;
        }

        return launch.OpName is "gather" or "memcopy" or "matmul";
    }

    private static CudaTritonKernelLaunch WithAttrs(CudaTritonKernelLaunch launch, Action<Dictionary<string, object?>> update)
    {
        var attrs = CloneAttrs(launch.OpAttributes);
        update(attrs);
        return launch with { OpAttributes = attrs };
    }

    private static Dictionary<string, object?> CloneAttrs(IReadOnlyDictionary<string, object?>? attrs)
        => attrs is null
            ? new Dictionary<string, object?>(StringComparer.Ordinal)
            : new Dictionary<string, object?>(attrs, StringComparer.Ordinal);

    private static IEnumerable<IReadOnlyDictionary<string, object?>> GetTailList(IReadOnlyDictionary<string, object?>? attrs)
    {
        if (attrs is null || !attrs.TryGetValue(CclTailAttrsKey, out var value) || value is not IEnumerable<IReadOnlyDictionary<string, object?>> tails)
        {
            return Array.Empty<IReadOnlyDictionary<string, object?>>();
        }

        return tails;
    }

    private static Dictionary<string, object?> CloneTail(IReadOnlyDictionary<string, object?> tail)
        => new(tail, StringComparer.Ordinal);

    private static Dictionary<string, object?> ToObjectMap(IReadOnlyDictionary<string, CudaTritonBufferDesc> buffers)
        => buffers.ToDictionary(kv => kv.Key, kv => (object?)kv.Value, StringComparer.Ordinal);

    private static string SanitizeName(string opName)
    {
        var value = opName.StartsWith("fusion.", StringComparison.Ordinal) ? opName["fusion.".Length..] : opName;
        value = Regex.Replace(value, "[^A-Za-z0-9_]+", "_");
        return value.Trim('_');
    }

    private sealed record ProducerRef(
        CudaTritonFunctionSource Function,
        int LaunchIndex,
        CudaTritonKernelLaunch Launch,
        BufferKey OutputKey,
        CudaTritonFunctionSource? ParentFunction = null,
        int? ParentFunctionLaunchIndex = null);

    private sealed record ProducerSiteKey(string FunctionName, int LaunchIndex)
    {
        public static ProducerSiteKey From(ProducerRef producer)
            => new(producer.Function.Name, producer.LaunchIndex);
    }

    private sealed record OutputBufferRef(int Index, string ArgumentName, CudaTritonBufferDesc? Desc);

    private sealed record BufferKey(
        string Name,
        string Location,
        int Hierarchy,
        string BufferSizeBytes,
        string SpanStartBytes,
        string SpanSizeBytes,
        string? BaseStart)
    {
        public static BufferKey From(CudaTritonBufferDesc desc)
        {
            var memory = desc.Memory;
            return new BufferKey(
                desc.Name,
                memory.Location,
                memory.Hierarchy,
                MemoryKey.DimKey(memory.BufferSizeBytes),
                MemoryKey.DimKey(memory.SpanStartBytes),
                MemoryKey.DimKey(memory.SpanSizeBytes),
                memory.BaseStart);
        }
    }

    private sealed record MemoryKey(
        string Location,
        int Hierarchy,
        string BufferSizeBytes,
        string SpanStartBytes,
        string SpanSizeBytes,
        string? BaseStart)
    {
        public static MemoryKey From(CudaTritonBufferDesc desc)
        {
            var memory = desc.Memory;
            return new MemoryKey(
                memory.Location,
                memory.Hierarchy,
                DimKey(memory.BufferSizeBytes),
                DimKey(memory.SpanStartBytes),
                DimKey(memory.SpanSizeBytes),
                memory.BaseStart);
        }

        public static string DimKey(CudaTritonDimDesc dim)
            => dim.Value.HasValue
                ? $"fixed:{dim.Value.Value.ToString(CultureInfo.InvariantCulture)}"
                : $"{dim.Kind}:{dim.Symbol ?? dim.Expression ?? string.Empty}";
    }
}
