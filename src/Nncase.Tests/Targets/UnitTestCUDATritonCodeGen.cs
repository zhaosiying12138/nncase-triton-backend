// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.Json;
using Nncase.CodeGen.NTT.CUDA;
using Nncase.IR;
using Nncase.Targets;
using Nncase.TIR;
using Xunit;
using NttF = Nncase.TIR.F.NTT;
using TIR = Nncase.TIR;

namespace Nncase.Tests.TargetTest;

public sealed class UnitTestCUDATritonCodeGen
{
    [Fact]
    public void TritonSourceContainsMetadataAndFunctionDispatch()
    {
        var module = new CudaTritonModuleSource(
            PeCount: 4,
            RdataPoolSize: 128,
            ThreadLocalRdataPoolSize: 64,
            BlockLocalRdataPoolSize: 32,
            Functions:
            [
                new CudaTritonFunctionSource(
                    Id: 7,
                    Name: "main",
                    IsEntry: true,
                    Parameters: ["x", "y"],
                    LocalDataPoolSize: 256,
                    OutputPoolSize: 512,
                    RdataPoolSize: 128,
                    Launches:
                    [
                        new CudaTritonKernelLaunch(0, CudaTritonLaunchKind.Compute, "elementwise.add", ["x", "y"], false),
                    ]),
            ]);

        var source = new TritonPythonSourceBuilder().Build(module);

        Assert.Contains("__nncase_triton_metadata__", source, StringComparison.Ordinal);
        Assert.Contains("\"pe_count\": 4", source, StringComparison.Ordinal);
        Assert.Contains("\"function_ids\": {\"main\": 7}", source, StringComparison.Ordinal);
        Assert.Contains("def launch(function_id, pe_id, data_pool, output_pool, rdata_pool, *function_args", source, StringComparison.Ordinal);
        Assert.Contains("if function_id == 7:", source, StringComparison.Ordinal);
        Assert.Contains("return invoke_main(pe_id, data_pool, output_pool, rdata_pool, *function_args", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonSourceExposesMultiPeLockstepCollectiveRuntime()
    {
        var source = new TritonPythonSourceBuilder().Build(new CudaTritonModuleSource(
            PeCount: 16,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions: []));

        Assert.Contains("all_data_pools=None", source, StringComparison.Ordinal);
        Assert.Contains("def _execute_multi_pe_function", source, StringComparison.Ordinal);
        Assert.Contains("def _run_gather_reduce_scatter_multi", source, StringComparison.Ordinal);
        Assert.Contains("def _local_shape_and_offsets", source, StringComparison.Ordinal);
        Assert.Contains("def _run_tensor_store_multi", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonSourceMaterializesPartialAliasesBeforeFullViewReads()
    {
        var source = new TritonPythonSourceBuilder().Build(new CudaTritonModuleSource(
            PeCount: 16,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions: []));

        Assert.Contains("def _memory_key", source, StringComparison.Ordinal);
        Assert.Contains("def _launch_output_indices", source, StringComparison.Ordinal);
        Assert.Contains("def _is_launch_output_desc", source, StringComparison.Ordinal);
        Assert.Contains("def _record_partial_aliases", source, StringComparison.Ordinal);
        Assert.Contains("def _materialize_partial_aliases", source, StringComparison.Ordinal);
        Assert.Contains("def _materialize_partial_inputs", source, StringComparison.Ordinal);
        Assert.Contains("context.setdefault(\"partial_aliases\", {})[key] = desc", source, StringComparison.Ordinal);
        Assert.Contains("and _is_launch_output_desc(launch_meta, key, desc)):", source, StringComparison.Ordinal);
        Assert.Contains("full = _materialize_global_tensor(contexts, partial_desc.get(\"name\"), partial_desc", source, StringComparison.Ordinal);
        Assert.Contains("target_shape = _append_vector_lanes_to_shape(_resolve_shape(contexts[0], desc), _desc_vector_lanes(desc, _desc_distributed_type(desc)))", source, StringComparison.Ordinal);
        Assert.Contains("if int(full.numel()) != int(math.prod(target_shape)):", source, StringComparison.Ordinal);
        Assert.Contains("full = full.reshape(tuple(target_shape))", source, StringComparison.Ordinal);
        Assert.Contains("_materialize_partial_aliases(contexts, launch_meta)", source, StringComparison.Ordinal);
        Assert.Contains("_materialize_partial_inputs(contexts, launch_meta)", source, StringComparison.Ordinal);
        Assert.Contains("_record_partial_aliases(contexts, launch_meta)", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonSourceSupportsVectorPackUnpackFallback()
    {
        var source = new TritonPythonSourceBuilder().Build(new CudaTritonModuleSource(
            PeCount: 16,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions: []));

        Assert.Contains("def _run_pack", source, StringComparison.Ordinal);
        Assert.Contains("def _run_unpack", source, StringComparison.Ordinal);
        Assert.Contains("def _desc_vector_lanes", source, StringComparison.Ordinal);
        Assert.Contains("_append_vector_lanes_to_shape(shape, lanes)", source, StringComparison.Ordinal);
        Assert.Contains("if op_name == \"pack\":", source, StringComparison.Ordinal);
        Assert.Contains("if op_name == \"unpack\":", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonSourceOffsetsShardedPositionIdsByPeLocalSlice()
    {
        var source = new TritonPythonSourceBuilder().Build(new CudaTritonModuleSource(
            PeCount: 16,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions: []));

        Assert.Contains("def _run_get_position_ids", source, StringComparison.Ordinal);
        Assert.Contains("desc = _find_launch_buffer_desc(context, args[1], context.get(\"current_launch_metadata\"), 1)", source, StringComparison.Ordinal);
        Assert.Contains("bound = context.get(\"bound_args\", {}).get(_strip_l0(args[1]))", source, StringComparison.Ordinal);
        Assert.Contains("desc = bound.desc", source, StringComparison.Ordinal);
        Assert.Contains("_, offsets = _local_shape_and_offsets(context, desc, global_shape, _desc_distributed_type(desc))", source, StringComparison.Ordinal);
        Assert.Contains("local_offset = int(offsets[int(shard_axes[0])])", source, StringComparison.Ordinal);
        Assert.Contains("torch.arange(start + local_offset, start + local_offset + out.numel()", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonSourceMergesVectorLanesForPagedAttentionLayoutFallback()
    {
        var source = new TritonPythonSourceBuilder().Build(new CudaTritonModuleSource(
            PeCount: 16,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions: []));

        Assert.Contains("def _merge_layout_vector_lanes", source, StringComparison.Ordinal);
        Assert.Contains("tensor = _merge_layout_vector_lanes(tensor, layout)", source, StringComparison.Ordinal);
        Assert.Contains("def _split_layout_vector_lanes", source, StringComparison.Ordinal);
        Assert.Contains("temp_out = _empty_like_global(out_desc", source, StringComparison.Ordinal);
        Assert.Contains("y = _from_hds(y, attrs.get(\"layout\"), out)", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonSourceSupportsConcatFallback()
    {
        var source = new TritonPythonSourceBuilder().Build(new CudaTritonModuleSource(
            PeCount: 16,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions: []));

        Assert.Contains("def _run_concat", source, StringComparison.Ordinal);
        Assert.Contains("def _infer_concat_axis", source, StringComparison.Ordinal);
        Assert.Contains("torch.cat([tensor.to(out.dtype) for tensor in inputs], dim=axis)", source, StringComparison.Ordinal);
        Assert.Contains("if op_name == \"concat\":", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonSourceSupportsVerboseLaunchTracing()
    {
        var source = new TritonPythonSourceBuilder().Build(new CudaTritonModuleSource(
            PeCount: 16,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions: []));

        Assert.Contains("NNCASE_TRITON_VERBOSE", source, StringComparison.Ordinal);
        Assert.Contains("NNCASE_TRITON_VERBOSES", source, StringComparison.Ordinal);
        Assert.Contains("NNCASE_CUDA_VERBOSE", source, StringComparison.Ordinal);
        Assert.Contains("NNCASE_TRITON_VERBOSE_LIMIT", source, StringComparison.Ordinal);
        Assert.Contains("launch<grid=", source, StringComparison.Ordinal);
        Assert.Contains("elapsed_ms=", source, StringComparison.Ordinal);
        Assert.Contains("def _execute_launch_body", source, StringComparison.Ordinal);
        Assert.Contains("def _execute_multi_pe_launch_body", source, StringComparison.Ordinal);
        Assert.Contains("_verbose_launch_begin(contexts, kind, op_name, argument_names, launch_meta)", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonSourceUsesByteWideDLPackBool()
    {
        var source = new TritonPythonSourceBuilder().Build(new CudaTritonModuleSource(
            PeCount: 1,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions: []));

        Assert.Contains("if dtype is torch.bool:", source, StringComparison.Ordinal);
        Assert.Contains("return _DLDataType(6, 8, 1)", source, StringComparison.Ordinal);
        Assert.DoesNotContain("return _DLDataType(6, 1, 1)", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonSourceTreatsUnmaterializedMatmulLoadCAsFalse()
    {
        var source = new TritonPythonSourceBuilder().Build(new CudaTritonModuleSource(
            PeCount: 1,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions: []));

        Assert.Contains("if len(args) > 3 and args[3] == \"call\":", source, StringComparison.Ordinal);
        Assert.Contains("load_c = False", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonSourceMaterializesMatmulOutputFromResultWhenDescriptorIsAbsent()
    {
        var source = new TritonPythonSourceBuilder().Build(new CudaTritonModuleSource(
            PeCount: 1,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions: []));

        Assert.Contains("except KeyError:", source, StringComparison.Ordinal);
        Assert.Contains("out = None", source, StringComparison.Ordinal);
        Assert.Contains("compute_dtype = result.dtype", source, StringComparison.Ordinal);
        Assert.Contains("return _write_tensor(context, args[2], result.to(compute_dtype))", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonSourceAppliesRopeOverLastDimensionForQwenLayout()
    {
        var source = new TritonPythonSourceBuilder().Build(new CudaTritonModuleSource(
            PeCount: 1,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions: []));

        Assert.Contains("head, seq, dim = x.shape", source, StringComparison.Ordinal);
        Assert.Contains("first = x[:, :, :half]", source, StringComparison.Ordinal);
        Assert.Contains("y[:, :, half:] = second * cos[:, half:].unsqueeze(0) + first * sin[:, half:].unsqueeze(0)", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonSourceDispatchesShapeBucketEntrySegments()
    {
        var source = new TritonPythonSourceBuilder().Build(new CudaTritonModuleSource(
            PeCount: 1,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions:
            [
                new CudaTritonFunctionSource(
                    Id: 0,
                    Name: "main_prim",
                    IsEntry: true,
                    Parameters: ["input_ids", "kvCache"],
                    LocalDataPoolSize: 0,
                    OutputPoolSize: 0,
                    RdataPoolSize: 0,
                    Launches:
                    [
                        new CudaTritonKernelLaunch(0, CudaTritonLaunchKind.Function, "main_segment_0_prim", ["input_ids", "kvCache", "buffer_0", "data_0"], false),
                        new CudaTritonKernelLaunch(1, CudaTritonLaunchKind.Function, "main_segment_1_prim", ["input_ids", "kvCache", "buffer_0", "data_1"], false),
                    ]),
                new CudaTritonFunctionSource(Id: 1, Name: "main_segment_0_prim", IsEntry: false, Parameters: ["input_ids", "kvCache", "out"], LocalDataPoolSize: 0, OutputPoolSize: 0, RdataPoolSize: 0, Launches: []),
                new CudaTritonFunctionSource(Id: 2, Name: "main_segment_1_prim", IsEntry: false, Parameters: ["input_ids", "kvCache", "out"], LocalDataPoolSize: 0, OutputPoolSize: 0, RdataPoolSize: 0, Launches: []),
            ]));

        Assert.Contains("def _select_shape_bucket_launch", source, StringComparison.Ordinal);
        Assert.Contains("_selected_shape_bucket_ordinal = _select_shape_bucket_launch(function_meta, [0, 1]", source, StringComparison.Ordinal);
        Assert.Contains("if _selected_shape_bucket_ordinal == 0:", source, StringComparison.Ordinal);
        Assert.Contains("if _selected_shape_bucket_ordinal == 1:", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonSourceUpdatesDenseKvCacheByCurrentTokenRange()
    {
        var source = new TritonPythonSourceBuilder().Build(new CudaTritonModuleSource(
            PeCount: 1,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions: []));

        Assert.Contains("start = int(_KV_STATE.get(\"current_start\", 0))", source, StringComparison.Ordinal);
        Assert.Contains("old[:, :, start:end].copy_(value.to(old.dtype))", source, StringComparison.Ordinal);
        Assert.Contains("k = k[:, :, :end]", source, StringComparison.Ordinal);
        Assert.DoesNotContain("torch.cat([old, value], dim=2)", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonSourceKeepsCollectiveAndReduceAsSeparateLaunches()
    {
        var module = new CudaTritonModuleSource(
            PeCount: 2,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions:
            [
                new CudaTritonFunctionSource(
                    Id: 0,
                    Name: "main",
                    IsEntry: true,
                    Parameters: ["x", "y", "z"],
                    LocalDataPoolSize: 0,
                    OutputPoolSize: 0,
                    RdataPoolSize: 0,
                    Launches:
                    [
                        new CudaTritonKernelLaunch(0, CudaTritonLaunchKind.Compute, "elementwise.mul", ["x", "y"], false),
                        new CudaTritonKernelLaunch(1, CudaTritonLaunchKind.Boxing, "tensor_load", ["x", "z"], true),
                        new CudaTritonKernelLaunch(2, CudaTritonLaunchKind.Reduce, "sum", ["z"], true),
                        new CudaTritonKernelLaunch(3, CudaTritonLaunchKind.Matmul, "matmul", ["x", "y", "z"], false),
                    ]),
            ]);

        var source = new TritonPythonSourceBuilder().Build(module);

        Assert.Contains("_launch_compute(\"elementwise.mul\"", source, StringComparison.Ordinal);
        Assert.Contains("_launch_boxing(\"tensor_load\"", source, StringComparison.Ordinal);
        Assert.Contains("_launch_reduce(\"sum\"", source, StringComparison.Ordinal);
        Assert.Contains("_launch_matmul(\"matmul\"", source, StringComparison.Ordinal);
        Assert.DoesNotContain("_launch_compute(\"tensor_load\"", source, StringComparison.Ordinal);
        Assert.DoesNotContain("_launch_compute(\"sum\"", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TritonMetadataSkipsCclScratchWhenNoCollectiveLaunches()
    {
        var module = new CudaTritonModuleSource(
            PeCount: 1,
            RdataPoolSize: 1024,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 2048,
            Functions:
            [
                new CudaTritonFunctionSource(
                    Id: 0,
                    Name: "main",
                    IsEntry: true,
                    Parameters: ["x"],
                    LocalDataPoolSize: 4096,
                    OutputPoolSize: 8192,
                    RdataPoolSize: 0,
                    Launches:
                    [
                        new CudaTritonKernelLaunch(0, CudaTritonLaunchKind.Matmul, "matmul", ["x", "y", "z"], false),
                    ]),
            ]);

        var metadata = new TritonPythonSourceBuilder().BuildMetadataJson(module);
        using var document = JsonDocument.Parse(metadata);

        Assert.Equal(0, document.RootElement.GetProperty("ccl_scratch_bytes").GetInt64());
    }

    [Fact]
    public void TritonMetadataSkipsCclScratchForPythonLockstepCollectives()
    {
        var module = new CudaTritonModuleSource(
            PeCount: 16,
            RdataPoolSize: 1024,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 2048,
            Functions:
            [
                new CudaTritonFunctionSource(
                    Id: 0,
                    Name: "main",
                    IsEntry: true,
                    Parameters: ["x", "y"],
                    LocalDataPoolSize: 4096,
                    OutputPoolSize: 8192,
                    RdataPoolSize: 0,
                    Launches:
                    [
                        new CudaTritonKernelLaunch(0, CudaTritonLaunchKind.Collective, "gather_reduce_scatter", ["x", "y"], true),
                    ]),
            ]);

        var metadata = new TritonPythonSourceBuilder().BuildMetadataJson(module);
        using var document = JsonDocument.Parse(metadata);

        Assert.Equal(0, document.RootElement.GetProperty("ccl_scratch_bytes").GetInt64());
    }

    [Fact]
    public void TritonMetadataPreservesBufferDescriptorsFromTirLaunches()
    {
        var input = CreateBuffer("input", DataTypes.Float32, MemoryLocation.Input, 16, 24, [2, 3], [3, 1]);
        var output = CreateBuffer("output", DataTypes.Float32, MemoryLocation.Output, 32, 8, [2], [1]);
        var body = new Sequential((Expr)NttF.Reduce(input, output, BoolConst(false), [1], [0], [1], true, ReduceOp.Sum));
        var function = new PrimFunction("main", "cuda", body, Array.Empty<IVar>()) { IsEntry = true };

        using var document = BuildMetadata(function);
        var functionJson = document.RootElement.GetProperty("functions")[0];
        var launch = functionJson.GetProperty("launches")[0];
        var inputDesc = launch.GetProperty("buffer_arguments").GetProperty("input");
        var outputDesc = functionJson.GetProperty("buffers").GetProperty("output");

        Assert.Equal("DataTypes.Float32", inputDesc.GetProperty("dtype").GetString());
        Assert.Equal(2, inputDesc.GetProperty("shape")[0].GetProperty("value").GetInt64());
        Assert.Equal(3, inputDesc.GetProperty("shape")[1].GetProperty("value").GetInt64());
        Assert.Equal(3, inputDesc.GetProperty("strides")[0].GetProperty("value").GetInt64());
        Assert.Equal(1, inputDesc.GetProperty("strides")[1].GetProperty("value").GetInt64());
        Assert.Equal("Input", inputDesc.GetProperty("memory").GetProperty("location").GetString());
        Assert.Equal(16, inputDesc.GetProperty("memory").GetProperty("alignment").GetInt32());
        Assert.Equal(24, inputDesc.GetProperty("memory").GetProperty("buffer_size_bytes").GetProperty("value").GetInt64());
        Assert.Equal("Output", outputDesc.GetProperty("memory").GetProperty("location").GetString());
        Assert.Equal(8, outputDesc.GetProperty("memory").GetProperty("span_size_bytes").GetProperty("value").GetInt64());
    }

    [Fact]
    public void TritonMetadataPreservesCommonOpAttributesFromTirLaunches()
    {
        var lhs = CreateBuffer("lhs", DataTypes.Float16, MemoryLocation.Input, 16, 64, [2, 4], [4, 1]);
        var rhs = CreateBuffer("rhs", DataTypes.Float16, MemoryLocation.Input, 16, 64, [4, 8], [8, 1]);
        var output = CreateBuffer("acc", DataTypes.Float32, MemoryLocation.Output, 16, 64, [2, 8], [8, 1]);
        var body = new Sequential(
            (Expr)NttF.Matmul(lhs, rhs, output, BoolConst(true), FloatConst(0.5f), None.Default, [1], [0], transA: true, fusedReduce: true, cSourcePath: "kernels/mm.py", funcName: "mm_kernel"));
        var function = new PrimFunction("main", "cuda", body, Array.Empty<IVar>()) { IsEntry = true };

        using var document = BuildMetadata(function);
        var launches = document.RootElement.GetProperty("functions")[0].GetProperty("launches");
        var matmulAttrs = launches[0].GetProperty("op_attrs");
        var fusedReduceLaunch = launches[1];

        Assert.True(matmulAttrs.GetProperty("transpose_a").GetBoolean());
        Assert.False(matmulAttrs.GetProperty("transpose_b").GetBoolean());
        Assert.True(matmulAttrs.GetProperty("fused_reduce").GetBoolean());
        Assert.Equal(1, matmulAttrs.GetProperty("lhs_vectorized_axes")[0].GetInt32());
        Assert.Equal(0, matmulAttrs.GetProperty("rhs_vectorized_axes")[0].GetInt32());
        Assert.Equal("kernels/mm.py", matmulAttrs.GetProperty("c_source_path").GetString());
        Assert.Equal("mm_kernel", matmulAttrs.GetProperty("func_name").GetString());
        Assert.Equal("matmul.fused_reduce", fusedReduceLaunch.GetProperty("op_name").GetString());
        Assert.True(fusedReduceLaunch.GetProperty("op_attrs").GetProperty("fused_reduce").GetBoolean());
    }

    [Fact]
    public void TritonMetadataPreservesPackUnpackAttributesFromTirLaunches()
    {
        var packedType = new VectorType(DataTypes.Float32, 8);
        var input = CreateBuffer("input", DataTypes.Float32, MemoryLocation.Input, 16, 2048, [1, 128, 4], [512, 4, 1]);
        var packed = CreateBuffer("packed", packedType, MemoryLocation.Data, 32, 2048, [1, 16, 4], [64, 4, 1]);
        var output = CreateBuffer("output", DataTypes.Float32, MemoryLocation.Output, 16, 2048, [1, 128, 4], [512, 4, 1]);
        var body = new Sequential(
            (Expr)NttF.Pack(input, packed, [8], [1]),
            (Expr)NttF.Unpack(packed, output, [8], [1]));
        var function = new PrimFunction("main", "cuda", body, Array.Empty<IVar>()) { IsEntry = true };

        using var document = BuildMetadata(function);
        var launches = document.RootElement.GetProperty("functions")[0].GetProperty("launches");
        var packAttrs = launches[0].GetProperty("op_attrs");
        var unpackAttrs = launches[1].GetProperty("op_attrs");

        Assert.Equal("pack", launches[0].GetProperty("op_name").GetString());
        Assert.Equal(8, packAttrs.GetProperty("lanes")[0].GetInt32());
        Assert.Equal(1, packAttrs.GetProperty("axes")[0].GetInt32());
        Assert.Equal("unpack", launches[1].GetProperty("op_name").GetString());
        Assert.Equal(8, unpackAttrs.GetProperty("lanes")[0].GetInt32());
        Assert.Equal(1, unpackAttrs.GetProperty("axes")[0].GetInt32());
    }

    [Fact]
    public void TritonMetadataPreservesConcatAxisFromTirLaunches()
    {
        var lhs = CreateBuffer("lhs", DataTypes.Float16, MemoryLocation.Input, 16, 16, [1, 2], [2, 1]);
        var rhs = CreateBuffer("rhs", DataTypes.Float16, MemoryLocation.Input, 16, 32, [2, 2], [2, 1]);
        var output = CreateBuffer("output", DataTypes.Float16, MemoryLocation.Output, 16, 48, [3, 2], [2, 1]);
        var body = new Sequential((Expr)NttF.Concat([lhs, rhs], output, 0));
        var function = new PrimFunction("main", "cuda", body, Array.Empty<IVar>()) { IsEntry = true };

        using var document = BuildMetadata(function);
        var launch = document.RootElement.GetProperty("functions")[0].GetProperty("launches")[0];
        var attrs = launch.GetProperty("op_attrs");

        Assert.Equal("concat", launch.GetProperty("op_name").GetString());
        Assert.Equal(0, attrs.GetProperty("axis").GetInt32());
    }

    [Fact]
    public void CudaCodeGenAcceptsAutoTilingDeviceFunctions()
    {
        var input = CreateBuffer("input", DataTypes.Float32, MemoryLocation.Input, 16, 16, [4], [1]);
        var output = CreateBuffer("output", DataTypes.Float32, MemoryLocation.Output, 16, 16, [4], [1]);
        var body = new Sequential(T.Memcopy(output, input));
        var function = new PrimFunction("device_func_0", "cuda", body, Array.Empty<IVar>()) { IsEntry = false };

        using var document = BuildMetadata(function);
        var functionJson = document.RootElement.GetProperty("functions")[0];

        Assert.Equal("device_func_0", functionJson.GetProperty("name").GetString());
        Assert.False(functionJson.GetProperty("is_entry").GetBoolean());
        Assert.Equal("memcopy", functionJson.GetProperty("launches")[0].GetProperty("op_name").GetString());
    }

    [Fact]
    public void TritonMetadataPreservesAllocateBufferViewArguments()
    {
        var input = CreateBuffer("input", DataTypes.Float32, MemoryLocation.Input, 16, 16, [4], [1]);
        var output = CreateBuffer("output", DataTypes.Float32, MemoryLocation.Output, 16, 16, [4], [1]);
        var body = new Sequential(T.Memcopy(output, Nncase.IR.F.Buffer.AllocateBufferView(input)));
        var function = new PrimFunction("device_func_0", "cuda", body, Array.Empty<IVar>()) { IsEntry = false };

        using var document = BuildMetadata(function);
        var launch = document.RootElement.GetProperty("functions")[0].GetProperty("launches")[0];
        var arguments = launch.GetProperty("arguments");
        var bufferArguments = launch.GetProperty("buffer_arguments");

        Assert.Equal("output", arguments[0].GetString());
        Assert.Equal("input", arguments[1].GetString());
        Assert.Equal("Input", bufferArguments.GetProperty("src").GetProperty("memory").GetProperty("location").GetString());
    }

    [Fact]
    public void TritonMetadataPreservesAllocateBufferViewOfFunctionParameters()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, [4]));
        var output = CreateBuffer("output", DataTypes.Float32, MemoryLocation.Output, 16, 16, [4], [1]);
        var body = new Sequential(T.Memcopy(output, Nncase.IR.F.Buffer.AllocateBufferView(input)));
        var function = new PrimFunction("device_func_0", "cuda", body, new IVar[] { input }) { IsEntry = false };

        using var document = BuildMetadata(function);
        var launch = document.RootElement.GetProperty("functions")[0].GetProperty("launches")[0];
        var arguments = launch.GetProperty("arguments");
        var bufferArguments = launch.GetProperty("buffer_arguments");

        Assert.Equal("output", arguments[0].GetString());
        Assert.Equal("input", arguments[1].GetString());
        Assert.Equal("Output", bufferArguments.GetProperty("dest").GetProperty("memory").GetProperty("location").GetString());
    }

    [Fact]
    public void TritonMetadataPreservesBufferSubviewBaseArguments()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(4)));
        var output = CreateBuffer("output", DataTypes.Float32, MemoryLocation.Output, 16, 16, [4], [1]);
        var body = new Sequential(T.Memcopy(output, Nncase.IR.F.Buffer.BufferSubview(input, new RankedShape(0), new RankedShape(4))));
        var function = new PrimFunction("device_func_0", "cuda", body, new IVar[] { input }) { IsEntry = false };

        using var document = BuildMetadata(function);
        var launch = document.RootElement.GetProperty("functions")[0].GetProperty("launches")[0];
        var arguments = launch.GetProperty("arguments");

        Assert.Equal("output", arguments[0].GetString());
        Assert.Equal("input", arguments[1].GetString());
    }

    [Fact]
    public void TritonMetadataSerializesOrderedReturnBuffers()
    {
        var output = CreateBuffer("output", DataTypes.Float32, MemoryLocation.Output, 16, 32, [2, 4], [4, 1]);
        var body = new Sequential(T.Return(output));
        var function = new PrimFunction("main", "cuda", body, Array.Empty<IVar>()) { IsEntry = true };

        using var document = BuildMetadata(function);
        var functionJson = document.RootElement.GetProperty("functions")[0];
        var returns = functionJson.GetProperty("returns");
        var return0 = returns[0];
        var buffer = return0.GetProperty("buffer");

        Assert.Equal(0, return0.GetProperty("index").GetInt32());
        Assert.Equal("output", return0.GetProperty("value").GetString());
        Assert.Equal("DataTypes.Float32", buffer.GetProperty("dtype").GetString());
        Assert.Equal("Output", buffer.GetProperty("memory").GetProperty("location").GetString());
        Assert.Equal(2, buffer.GetProperty("shape")[0].GetProperty("value").GetInt64());
        Assert.Equal(4, buffer.GetProperty("shape")[1].GetProperty("value").GetInt64());
    }

    [Fact]
    public void TritonMetadataSerializesNestedFunctionCallsAsLaunches()
    {
        var input = CreateBuffer("input", DataTypes.Float32, MemoryLocation.Input, 16, 16, [4], [1]);
        var output = CreateBuffer("output", DataTypes.Float32, MemoryLocation.Output, 16, 16, [4], [1]);
        var segment = new PrimFunction("main_segment_0_prim", "cuda", new Sequential(T.Memcopy(output, input)), Array.Empty<IVar>()) { IsEntry = false };
        var main = new PrimFunction("main_prim", "cuda", new Sequential(new Call(segment, input, output), T.Return(output)), Array.Empty<IVar>()) { IsEntry = true };

        using var document = BuildMetadata([main, segment]);
        var mainJson = document.RootElement.GetProperty("functions")[0];
        var launch = mainJson.GetProperty("launches")[0];

        Assert.Equal("function", launch.GetProperty("kind").GetString());
        Assert.Equal("main_segment_0_prim", launch.GetProperty("op_name").GetString());
        Assert.Equal("input", launch.GetProperty("arguments")[0].GetString());
        Assert.Equal("output", launch.GetProperty("arguments")[1].GetString());
    }

    [Fact]
    public void TritonSourceFiltersBufferizeOnlyNestedFunctionArguments()
    {
        var module = new CudaTritonModuleSource(
            PeCount: 1,
            RdataPoolSize: 0,
            ThreadLocalRdataPoolSize: 0,
            BlockLocalRdataPoolSize: 0,
            Functions:
            [
                new CudaTritonFunctionSource(
                    Id: 0,
                    Name: "main",
                    IsEntry: true,
                    Parameters: ["x"],
                    LocalDataPoolSize: 0,
                    OutputPoolSize: 0,
                    RdataPoolSize: 0,
                    Launches:
                    [
                        new CudaTritonKernelLaunch(0, CudaTritonLaunchKind.Function, "segment", ["x", "dim_var", "out", "data_0", "block_local_data_1"], false),
                    ]),
                new CudaTritonFunctionSource(
                    Id: 1,
                    Name: "segment",
                    IsEntry: false,
                    Parameters: ["x", "out"],
                    LocalDataPoolSize: 0,
                    OutputPoolSize: 0,
                    RdataPoolSize: 0,
                    Launches: []),
            ]);

        var source = new TritonPythonSourceBuilder().Build(module);

        Assert.Contains("def _filter_function_argument_names", source, StringComparison.Ordinal);
        Assert.Contains("name != \"dim_var\"", source, StringComparison.Ordinal);
        Assert.Contains("not name.startswith(\"data_\")", source, StringComparison.Ordinal);
        Assert.Contains("not name.startswith(\"block_local_data_\")", source, StringComparison.Ordinal);
        Assert.Contains("filtered_names = _filter_function_argument_names(argument_names)", source, StringComparison.Ordinal);
    }

    [Fact]
    public void LaunchSummaryListsFunctionPoolsAndLaunches()
    {
        var module = new CudaTritonModuleSource(
            PeCount: 8,
            RdataPoolSize: 1024,
            ThreadLocalRdataPoolSize: 64,
            BlockLocalRdataPoolSize: 32,
            Functions:
            [
                new CudaTritonFunctionSource(
                    Id: 3,
                    Name: "qwen_attn",
                    IsEntry: true,
                    Parameters: ["q", "k", "v"],
                    LocalDataPoolSize: 256,
                    OutputPoolSize: 128,
                    RdataPoolSize: 64,
                    BlockLocalDataPoolSize: 16,
                    Launches:
                    [
                        new CudaTritonKernelLaunch(1, CudaTritonLaunchKind.Matmul, "qwen.matmul", ["q", "k", "v"], true),
                    ]),
                new CudaTritonFunctionSource(
                    Id: 4,
                    Name: "qwen_reduce",
                    IsEntry: false,
                    Parameters: ["acc"],
                    LocalDataPoolSize: 512,
                    OutputPoolSize: 0,
                    RdataPoolSize: 0,
                    Launches:
                    [
                        new CudaTritonKernelLaunch(0, CudaTritonLaunchKind.Collective, "all_reduce", ["acc"], false),
                    ]),
            ]);

        var linkableModuleType = typeof(CudaModuleBuilder).Assembly.GetType("Nncase.CodeGen.NTT.CUDA.CudaLinkableModule");
        Assert.NotNull(linkableModuleType);
        var buildLaunchSummary = linkableModuleType.GetMethod("BuildLaunchSummary", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(buildLaunchSummary);

        var summary = Assert.IsType<string>(buildLaunchSummary.Invoke(null, [module]));

        Assert.Contains("module pe_count=8 rdata_pool_size=1024 thread_local_rdata_pool_size=64 block_local_rdata_pool_size=32", summary, StringComparison.Ordinal);
        Assert.Contains("function id=3 name=qwen_attn is_entry=true data_pool_size=256 output_pool_size=128 rdata_pool_size=64 block_local_data_pool_size=16", summary, StringComparison.Ordinal);
        Assert.Contains("launch ordinal=1 kind=matmul op_name=qwen.matmul arguments=[q, k, v] requires_collective=true", summary, StringComparison.Ordinal);
        Assert.Contains("function id=4 name=qwen_reduce is_entry=false data_pool_size=512 output_pool_size=0 rdata_pool_size=0 block_local_data_pool_size=0", summary, StringComparison.Ordinal);
        Assert.Contains("launch ordinal=0 kind=collective op_name=all_reduce arguments=[acc] requires_collective=false", summary, StringComparison.Ordinal);
    }

    private static TIR.Buffer CreateBuffer(
        string name,
        DataType dtype,
        MemoryLocation location,
        int alignment,
        Dimension sizeBytes,
        Dimension[] shape,
        Dimension[] strides)
    {
        var physical = new PhysicalBuffer(alignment, sizeBytes, location);
        return new TIR.Buffer(name, dtype, new MemSpan(physical), shape, strides, null);
    }

    private static TensorConst BoolConst(bool value) => Const.FromTensor(Tensor.FromScalar(value));

    private static TensorConst FloatConst(float value) => Const.FromTensor(Tensor.FromScalar(value));

    private static JsonDocument BuildMetadata(params PrimFunction[] functions)
    {
        var builder = new CudaModuleBuilder(new CompileOptions { TargetOptions = new NTTTargetOptions() });
        var linkedModule = builder.Build(functions).Link(null!);
        var section = linkedModule.Sections.Single(s => s.Name == ".cuda.meta");
        using var stream = new MemoryStream();
        section.Serialize(stream);
        return JsonDocument.Parse(Encoding.UTF8.GetString(stream.ToArray()));
    }
}
