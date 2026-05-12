// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using Nncase.Diagnostics;
using Nncase.Targets;
using Nncase.Utilities;

namespace Nncase.CodeGen.NTT.CUDA;

internal sealed class CudaLinkableModule : ILinkableModule
{
    private const int TextAlignment = 8;

    private readonly Stream _rdata;
    private readonly IReadOnlyList<Stream> _threadLocalRdatas;
    private readonly IReadOnlyList<Stream> _blockLocalRdatas;
    private readonly IReadOnlyList<CudaLinkableFunction> _functions;
    private readonly NTTTargetOptions _targetOptions;

    public CudaLinkableModule(
        Stream rdata,
        IReadOnlyList<Stream> threadLocalRdatas,
        IReadOnlyList<Stream> blockLocalRdatas,
        IReadOnlyList<CudaLinkableFunction> functions,
        NTTTargetOptions targetOptions)
    {
        _rdata = rdata;
        _threadLocalRdatas = threadLocalRdatas;
        _blockLocalRdatas = blockLocalRdatas;
        _functions = functions;
        _targetOptions = targetOptions;
    }

    public IReadOnlyList<ILinkableFunction> PublicFunctions => _functions;

    public ILinkedModule Link(ILinkContext linkContext)
    {
        var linkedFunctions = new List<LinkedFunction>();
        var text = new MemoryStream();
        using (var writer = new BinaryWriter(text, Encoding.UTF8, leaveOpen: true))
        {
            foreach (var function in _functions)
            {
                writer.Flush();
                writer.AlignPosition(TextAlignment);
                var textBegin = writer.Position();
                function.Text.Position = 0;
                function.Text.CopyTo(writer.BaseStream);
                linkedFunctions.Add(new LinkedFunction(
                    function.Id,
                    function.SourceFunction,
                    (ulong)textBegin,
                    (ulong)function.Text.Length,
                    function.Sections));
            }
        }

        var moduleSource = CudaComputeCclTailPlanner.Plan(CreateModuleSource());
        var source = new TritonPythonSourceBuilder().Build(moduleSource);
        var moduleMeta = new TritonPythonSourceBuilder().BuildMetadataJson(moduleSource);
        if (DumpScope.Current.IsEnabled(DumpFlags.CodeGen))
        {
            DumpCodeGenArtifacts(source, moduleMeta, moduleSource);
        }

        var rdataAlign = GetRdataAlignment();
        var sections = new List<ILinkedSection>
        {
            CreateTextSection(Encoding.UTF8.GetBytes(moduleMeta), CudaSectionNames.CudaMeta, 8),
            CreateTextSection(Encoding.UTF8.GetBytes("nncase.cuda.triton\n"), CudaSectionNames.TritonModule, 1),
            CreateTextSection(Encoding.UTF8.GetBytes(source), CudaSectionNames.TritonSource, 1),
            new LinkedSection(text, WellknownSectionNames.Text, 0, 8, (ulong)text.Length),
            new LinkedSection(_rdata, WellknownSectionNames.Rdata, 0, rdataAlign, (ulong)_rdata.Length),
            new LinkedMultipleContentsSection(_threadLocalRdatas, WellknownSectionNames.ThreadLocalRdata, 0, rdataAlign),
            new LinkedMultipleContentsSection(_blockLocalRdatas, WellknownSectionNames.BlockLocalRdata, 0, rdataAlign),
        };

        return new CudaLinkedModule(linkedFunctions, sections);
    }

    internal static string BuildLaunchSummary(CudaTritonModuleSource moduleSource)
    {
        var sb = new StringBuilder();
        sb.AppendLine(CultureInfo.InvariantCulture, $"module pe_count={moduleSource.PeCount} fused_kernel={NTTTargetOptions.FormatCudaFusedKernelMode(moduleSource.FusedKernelMode)} rdata_pool_size={moduleSource.RdataPoolSize} thread_local_rdata_pool_size={moduleSource.ThreadLocalRdataPoolSize} block_local_rdata_pool_size={moduleSource.BlockLocalRdataPoolSize}");
        foreach (var function in moduleSource.Functions)
        {
            sb.AppendLine(CultureInfo.InvariantCulture, $"function id={function.Id} name={EscapeSummaryValue(function.Name)} is_entry={FormatBool(function.IsEntry)} data_pool_size={function.LocalDataPoolSize} output_pool_size={function.OutputPoolSize} rdata_pool_size={function.RdataPoolSize} block_local_data_pool_size={function.BlockLocalDataPoolSize}");
            foreach (var launch in function.Launches.OrderBy(l => l.Ordinal))
            {
                var arguments = string.Join(", ", launch.Arguments.Select(EscapeSummaryValue));
                sb.AppendLine(CultureInfo.InvariantCulture, $"  launch ordinal={launch.Ordinal} kind={FormatLaunchKind(launch.Kind)} op_name={EscapeSummaryValue(launch.OpName)} arguments=[{arguments}] requires_collective={FormatBool(launch.RequiresCollective)}{FormatCclTailSummary(launch)}");
            }

            foreach (var ret in function.Returns ?? Array.Empty<CudaTritonReturnDesc>())
            {
                sb.AppendLine(CultureInfo.InvariantCulture, $"  return index={ret.Index} value={EscapeSummaryValue(ret.Value)} buffer={EscapeSummaryValue(ret.Buffer.Name)}");
            }
        }

        return sb.ToString();
    }

    private static void DumpCodeGenArtifacts(string source, string moduleMeta, CudaTritonModuleSource moduleSource)
    {
        var codegenDir = Path.Join(DumpScope.Current.Directory, "CodeGen", "cuda");
        if (!Directory.Exists(codegenDir))
        {
            Directory.CreateDirectory(codegenDir);
        }

        File.WriteAllText(Path.Join(codegenDir, "triton_module.py"), source, Encoding.UTF8);
        File.WriteAllText(Path.Join(codegenDir, "cuda_meta.json"), moduleMeta, Encoding.UTF8);
        File.WriteAllText(Path.Join(codegenDir, "launch_summary.txt"), BuildLaunchSummary(moduleSource), Encoding.UTF8);
    }

    private static string FormatBool(bool value) => value ? "true" : "false";

    private static string FormatCclTailSummary(CudaTritonKernelLaunch launch)
    {
        if (launch.OpAttributes is null)
        {
            return string.Empty;
        }

        var parts = new List<string>();
        if (launch.OpAttributes.TryGetValue("elided_by_ccl_tail", out var elided) && elided is true)
        {
            parts.Add("elided_by_ccl_tail=true");
        }

        if (launch.OpAttributes.TryGetValue("ccl_tail_unfused_reason", out var reason) && reason is string reasonText)
        {
            parts.Add($"ccl_tail_unfused_reason={EscapeSummaryValue(reasonText)}");
        }

        if (launch.OpAttributes.TryGetValue("ccl_tails", out var tails) && tails is System.Collections.IEnumerable enumerable)
        {
            foreach (var tail in enumerable)
            {
                if (tail is IReadOnlyDictionary<string, object?> map && map.TryGetValue("op_name", out var opName) && opName is string opNameText)
                {
                    parts.Add($"ccl_tail={EscapeSummaryValue(opNameText)}");
                }
            }
        }

        return parts.Count == 0 ? string.Empty : $" {string.Join(" ", parts)}";
    }

    private static string FormatLaunchKind(CudaTritonLaunchKind kind) => kind switch
    {
        CudaTritonLaunchKind.Compute => "compute",
        CudaTritonLaunchKind.Memcopy => "memcopy",
        CudaTritonLaunchKind.Boxing => "boxing",
        CudaTritonLaunchKind.Reduce => "reduce",
        CudaTritonLaunchKind.Matmul => "matmul",
        CudaTritonLaunchKind.Collective => "collective",
        CudaTritonLaunchKind.Function => "function",
        _ => kind.ToString(),
    };

    private static string EscapeSummaryValue(string value)
        => value
            .Replace("\\", "\\\\", StringComparison.Ordinal)
            .Replace("\r", "\\r", StringComparison.Ordinal)
            .Replace("\n", "\\n", StringComparison.Ordinal);

    private CudaTritonModuleSource CreateModuleSource()
    {
        var rdataPoolSize = _functions.Count == 0 ? 0 : _functions.Max(f => f.MemoryPoolDesc.RdataPoolSize);
        var threadLocalRdataPoolSize = _functions.Count == 0 ? 0 : _functions.Max(f => f.MemoryPoolDesc.ThreadLocalRdataPoolSize);
        var blockLocalRdataPoolSize = _functions.Count == 0 ? 0 : _functions.Max(f => f.MemoryPoolDesc.BlockLocalRdataPoolSize);
        return new CudaTritonModuleSource(
            TensorUtilities.GetProduct(_targetOptions.Hierarchies[0]),
            rdataPoolSize,
            threadLocalRdataPoolSize,
            blockLocalRdataPoolSize,
            _functions.Select(f => f.FunctionSource).ToArray(),
            _targetOptions.FusedKernelMode);
    }

    private uint GetRdataAlignment()
    {
        ulong rdataAlign = 8;
        foreach (var function in _functions)
        {
            rdataAlign = Math.Max(rdataAlign, function.SourceFunction.SchedResult.DataAlign);
        }

        return checked((uint)rdataAlign);
    }

    private LinkedSection CreateTextSection(byte[] bytes, string name, uint alignment)
    {
        var stream = new MemoryStream(bytes);
        return new LinkedSection(stream, name, 0, alignment, (ulong)stream.Length);
    }
}
