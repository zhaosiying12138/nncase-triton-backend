from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "modules" / "Nncase.Modules.NTT" / "CodeGen" / "CUDA" / "TritonPythonSourceBuilder.cs"
RUNTIME_FUNCTION = ROOT / "src" / "Native" / "src" / "runtime" / "cuda" / "runtime_function.cpp"
RUNTIME_MODULE = ROOT / "src" / "Native" / "src" / "runtime" / "cuda" / "runtime_module.cpp"


LINEAR_PE_KERNELS = (
    "_nncase_pe_copy_rank4_kernel",
    "_nncase_pe_unary_rank4_kernel",
    "_nncase_pe_binary_rank4_kernel",
    "_nncase_pe_swish_mul_rank4_kernel",
    "_nncase_pe_where_rank4_kernel",
    "_nncase_pe_concat2_rank4_kernel",
    "_nncase_pe_transpose_rank4_kernel",
    "_nncase_pe_rope_rank4_kernel",
    "_nncase_pe_position_ids_kernel",
    "_nncase_pe_update_kv_rank4_kernel",
    "_nncase_pe_gather_axis0_rank2_kernel",
)

ROW_PE_KERNELS = (
    "_nncase_pe_layer_norm_kernel",
    "_nncase_pe_layer_norm_transpose_rank4_kernel",
)

MATRIX_PE_KERNELS = (
    "_nncase_pe_matmul_kernel",
    "_nncase_pe_matmul_swish_mul_kernel",
    "_nncase_pe_silu_mul_matmul_kernel",
    "_nncase_pe_matmul_mul_matmul_kernel",
    "_nncase_pe_flash_attention_rank4_kernel",
)

PAGED_ATTENTION_KERNEL = "_nncase_paged_flash_attention_rank3_kernel"

CCL_KERNELS = (
    "_nncase_ccl_rank4_kernel",
    "_nncase_ccl_linear_rank4_kernel",
)

INLINE_CCL_TAIL_KERNELS = (
    "_nncase_pe_memcopy_grs_tail_kernel",
    "_nncase_pe_gather_grs_tail_kernel",
    "_nncase_pe_matmul_grs_tail_kernel",
)


def test_linear_pe_kernels_launch_through_persistent_pe_grid():
    source = SOURCE.read_text(encoding="utf-8")

    assert "def _launch_persistent_tile_kernel(contexts, kernel, kernel_name, block, meta, *args, **kwargs):" in source
    for kernel in LINEAR_PE_KERNELS:
        assert f"_launch_persistent_tile_kernel(contexts, {kernel}" in source
        assert f"_launch_triton_kernel(contexts, {kernel}" not in source


def test_linear_pe_kernels_loop_tiles_inside_pe_program():
    source = SOURCE.read_text(encoding="utf-8")

    for kernel in LINEAR_PE_KERNELS:
        start = source.index(f"def {kernel}")
        next_kernel = source.find("@triton.jit", start + 1)
        body = source[start: next_kernel if next_kernel != -1 else len(source)]
        assert "pe = tl.program_id(0)" in body
        assert "for tile in range(0, MAX_TILES):" in body
        assert "MAX_TILES:tl.constexpr" in body


def test_row_pe_kernels_launch_through_persistent_pe_grid():
    source = SOURCE.read_text(encoding="utf-8")

    for kernel in ROW_PE_KERNELS:
        assert f"_launch_persistent_tile_kernel(contexts, {kernel}" in source
        assert f"_launch_triton_kernel(contexts, {kernel}" not in source


def test_row_pe_kernels_loop_rows_inside_pe_program():
    source = SOURCE.read_text(encoding="utf-8")

    for kernel in ROW_PE_KERNELS:
        start = source.index(f"def {kernel}")
        next_kernel = source.find("@triton.jit", start + 1)
        body = source[start: next_kernel if next_kernel != -1 else len(source)]
        assert "pe = tl.program_id(0)" in body
        assert "for row in range(0, MAX_ROWS):" in body
        assert "MAX_ROWS:tl.constexpr" in body


def test_matrix_pe_kernels_launch_through_persistent_pe_grid():
    source = SOURCE.read_text(encoding="utf-8")

    for kernel in MATRIX_PE_KERNELS:
        assert f"_launch_persistent_tile_kernel(contexts, {kernel}" in source
        assert f"_launch_triton_kernel(contexts, {kernel}" not in source


def test_matrix_pe_kernels_loop_tiles_inside_pe_program():
    source = SOURCE.read_text(encoding="utf-8")

    for kernel in MATRIX_PE_KERNELS:
        start = source.index(f"def {kernel}")
        next_kernel = source.find("@triton.jit", start + 1)
        body = source[start: next_kernel if next_kernel != -1 else len(source)]
        assert "pe = tl.program_id(0)" in body
        assert "for pid_m in range(0, MAX_M_TILES):" in body
        assert "for pid_n in range(0, MAX_N_TILES):" in body
        assert "MAX_M_TILES:tl.constexpr" in body
        assert "MAX_N_TILES:tl.constexpr" in body


def test_flash_attention_loops_query_and_key_tiles_inside_pe_program():
    source = SOURCE.read_text(encoding="utf-8")

    kernel = "_nncase_pe_flash_attention_rank4_kernel"
    start = source.index(f"def {kernel}")
    next_kernel = source.find("@triton.jit", start + 1)
    body = source[start: next_kernel if next_kernel != -1 else len(source)]
    assert "pe = tl.program_id(0)" in body
    assert "for bh in range(0, MAX_BH):" in body
    assert "for pid_m in range(0, MAX_M_TILES):" in body
    assert "for pid_n in range(0, MAX_N_TILES):" in body
    assert "MAX_BH:tl.constexpr" in body
    assert "MAX_M_TILES:tl.constexpr" in body
    assert "MAX_N_TILES:tl.constexpr" in body


def test_paged_attention_launches_through_persistent_pe_grid():
    source = SOURCE.read_text(encoding="utf-8")

    assert f"_launch_persistent_tile_kernel(contexts, {PAGED_ATTENTION_KERNEL}" in source
    assert f"_launch_triton_kernel(contexts, {PAGED_ATTENTION_KERNEL}" not in source


def test_paged_attention_loops_query_work_inside_pe_program():
    source = SOURCE.read_text(encoding="utf-8")

    start = source.index(f"def {PAGED_ATTENTION_KERNEL}")
    next_kernel = source.find("@triton.jit", start + 1)
    body = source[start: next_kernel if next_kernel != -1 else len(source)]
    assert "pe = tl.program_id(0)" in body
    assert "for q_head in range(0, MAX_HEADS):" in body
    assert "for pid_m in range(0, MAX_Q_TILES):" in body
    assert "for pid_t in range(0, MAX_KV_TILES):" in body
    assert "m_next = tl.maximum(m_i, tl.max(scores, axis=1))" in body
    assert "acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)" in body
    assert "MAX_HEADS:tl.constexpr" in body
    assert "MAX_Q_TILES:tl.constexpr" in body
    assert "MAX_KV_TILES:tl.constexpr" in body


def test_ccl_kernels_launch_through_persistent_pe_grid():
    source = SOURCE.read_text(encoding="utf-8")

    for kernel in CCL_KERNELS:
        assert f"_launch_persistent_tile_kernel(contexts, {kernel}" in source
        assert f"_launch_triton_kernel(contexts, {kernel}" not in source


def test_ccl_kernels_loop_tiles_inside_pe_program():
    source = SOURCE.read_text(encoding="utf-8")

    for kernel in CCL_KERNELS:
        target = "_nncase_ccl_rank4_body" if kernel == "_nncase_ccl_rank4_kernel" else kernel
        start = source.index(f"def {target}")
        next_kernel = source.find("@triton.jit", start + 1)
        body = source[start: next_kernel if next_kernel != -1 else len(source)]
        if target == kernel:
            assert "pe = tl.program_id(0)" in body
        assert "for tile in range(0, MAX_TILES):" in body
        assert "MAX_TILES:tl.constexpr" in body


def test_compute_ccl_tail_kernels_inline_barrier_and_ccl_body():
    source = SOURCE.read_text(encoding="utf-8")

    assert "def _nncase_ccl_rank4_body(" in source
    assert "def _nncase_ccl_tail_barrier(" in source
    assert "tl.atomic_add(counter_ptr, 1, sem=\"acq_rel\")" in source
    assert "tl.atomic_add(generation_ptr, 0, sem=\"acquire\")" in source
    assert "tl.atomic_add(generation_ptr, 1, sem=\"release\")" in source
    for kernel in INLINE_CCL_TAIL_KERNELS:
        assert f"_launch_persistent_tile_kernel(contexts, {kernel}" in source
        start = source.index(f"def {kernel}")
        next_kernel = source.find("@triton.jit", start + 1)
        body = source[start: next_kernel if next_kernel != -1 else len(source)]
        assert "pe = tl.program_id(0)" in body
        assert "_nncase_ccl_tail_barrier(scratch_ptr, ccl_pe_count)" in body
        assert "_nncase_ccl_rank4_body(" in body


def test_compute_ccl_tail_runtime_does_not_launch_separate_host_tail():
    source = SOURCE.read_text(encoding="utf-8")

    assert "callee_context[\"parent_context\"] = context" in source
    assert "def _inline_ccl_tail_state_context_groups(contexts):" in source
    assert "context.setdefault(\"inline_ccl_tail_prepared\", {})[site] = prepared_tail" in source
    assert "def _function_has_inline_ccl_tails(function_meta):" in source
    assert "if allow_ccl_fusion and _function_has_inline_ccl_tails(callee_meta):" in source

    start = source.index("def _run_fused_ccl_tail(")
    end = source.index("def _run_fused_ccl_tails(", start)
    body = source[start:end]
    assert "_run_gather_reduce_scatter_ccl_triton" not in body
    assert "_run_ccl_tail_barrier" not in body
    assert "_finish_inline_ccl_tail(contexts, tail_meta, prepared_tail)" in body
    assert "inline CCL tail" in body


def test_partial_add_ccl_shortcut_is_not_registered():
    source = SOURCE.read_text(encoding="utf-8")

    assert "_nncase_partial_add_rank4_kernel" not in source
    assert "_run_partial_add_ccl_rank4_desc_triton" not in source
    assert '"fused_compute_ccl": True' not in source


def test_runtime_plumbs_ccl_scratch_to_generated_python_context():
    source = SOURCE.read_text(encoding="utf-8")
    runtime_function = RUNTIME_FUNCTION.read_text(encoding="utf-8")
    runtime_module = RUNTIME_MODULE.read_text(encoding="utf-8")

    assert '\\"ccl_scratch_pool\\": int(ccl_scratch_pool or 0)' in source
    assert '\\"ccl_scratch_bytes\\": int(ccl_scratch_bytes or 0)' in source
    assert "ccl_scratch_pool=ccl_scratch_pool, ccl_scratch_bytes=ccl_scratch_bytes" in source
    assert "def _ccl_scratch_layout(contexts, block):" not in source
    assert "payload_bytes = int(pe_count) * int(block) * 4 * 2" not in source
    assert "native CCL scratch requires" not in source
    assert "scratch_layout = _ccl_scratch_layout(contexts, block)" not in source
    assert "var cclScratchBytes = CudaComputeCclTailPlanner.GetScratchBytes(module);" in source
    assert "EstimateCclScratchBytes" not in source
    assert "const ulong maxCclBlockElements = 1024;" not in source
    assert "const ulong pingPongSlots = 2;" not in source
    assert "LaunchNeedsNativeCclScratch" not in source
    assert "scratch_tensor = _tensor_from_pointer(scratch_pool + offset, _require_torch().int64, (2,), (1,))" in source
    assert "scratch_pool + offset + 8" not in source
    assert "try_var(ccl_scratch, module().ccl_scratch(&ccl_scratch_bytes));" in runtime_function
    assert 'PyDict_SetItemString(kwargs.get(), "ccl_scratch_pool"' in runtime_module
    assert 'PyDict_SetItemString(kwargs.get(), "ccl_scratch_bytes"' in runtime_module
    assert 'cuMemsetD8(ccl_scratch_, 0, ccl_scratch_size_)' in runtime_module
