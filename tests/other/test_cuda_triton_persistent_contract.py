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
    "_nncase_partial_add_rank4_kernel",
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
        start = source.index(f"def {kernel}")
        next_kernel = source.find("@triton.jit", start + 1)
        body = source[start: next_kernel if next_kernel != -1 else len(source)]
        assert "pe = tl.program_id(0)" in body
        assert "for tile in range(0, MAX_TILES):" in body
        assert "MAX_TILES:tl.constexpr" in body


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
    assert "var cclScratchBytes = 0UL;" in source
    assert "EstimateCclScratchBytes" not in source
    assert "const ulong maxCclBlockElements = 1024;" not in source
    assert "const ulong pingPongSlots = 2;" not in source
    assert "LaunchNeedsNativeCclScratch" not in source
    assert "try_var(ccl_scratch, module().ccl_scratch(&ccl_scratch_bytes));" in runtime_function
    assert 'PyDict_SetItemString(kwargs.get(), "ccl_scratch_pool"' in runtime_module
    assert 'PyDict_SetItemString(kwargs.get(), "ccl_scratch_bytes"' in runtime_module
    assert 'cuMemsetD8(ccl_scratch_, 0, ccl_scratch_size_)' in runtime_module
