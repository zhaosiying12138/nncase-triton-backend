import pytest


def get_cuda_qwen_admission_runner():
    try:
        from huggingface_test_runner import CudaQwenAdmissionRunner
    except (ModuleNotFoundError, RuntimeError) as ex:
        pytest.skip(f"nncase runtime is not importable: {ex}")
    return CudaQwenAdmissionRunner


def test_cuda_pe_candidates_start_from_sm_count_and_downgrade():
    CudaQwenAdmissionRunner = get_cuda_qwen_admission_runner()

    assert CudaQwenAdmissionRunner.pe_candidates(80) == [80, 40, 20, 10, 5, 2, 1]
    assert CudaQwenAdmissionRunner.pe_candidates(1) == [1]


def test_cuda_pe_target_options_are_numa_per_candidate(monkeypatch):
    CudaQwenAdmissionRunner = get_cuda_qwen_admission_runner()

    monkeypatch.setenv("NNCASE_CUDA_SM_COUNT", "8")
    runner = CudaQwenAdmissionRunner("cuda_qwen_admission_options")

    options = runner.make_cuda_pe_target_options(4)

    assert options is not None


def test_cuda_required_pe_count_is_read_from_env(monkeypatch):
    CudaQwenAdmissionRunner = get_cuda_qwen_admission_runner()

    monkeypatch.setenv("NNCASE_CUDA_SM_COUNT", "8")
    monkeypatch.setenv("NNCASE_CUDA_REQUIRED_PE", "16")
    runner = CudaQwenAdmissionRunner("cuda_qwen_admission_required_pe")

    assert runner.cuda_required_pe_count() == 16


def test_cuda_fused_kernel_mode_defaults_to_off(monkeypatch):
    CudaQwenAdmissionRunner = get_cuda_qwen_admission_runner()

    monkeypatch.delenv("NNCASE_CUDA_FUSED_KERNEL", raising=False)
    runner = CudaQwenAdmissionRunner("cuda_qwen_admission_fused_default")

    assert runner.cuda_fused_kernel_mode() == "off"


def test_cuda_fused_kernel_mode_accepts_compute(monkeypatch):
    CudaQwenAdmissionRunner = get_cuda_qwen_admission_runner()

    monkeypatch.setenv("NNCASE_CUDA_FUSED_KERNEL", "compute")
    runner = CudaQwenAdmissionRunner("cuda_qwen_admission_fused_compute")

    assert runner.cuda_fused_kernel_mode() == "compute"


def test_cuda_fused_kernel_mode_rejects_tile_as_a_mode(monkeypatch):
    CudaQwenAdmissionRunner = get_cuda_qwen_admission_runner()
    import huggingface_test_runner

    monkeypatch.setenv("NNCASE_CUDA_FUSED_KERNEL", "tile")
    runner = CudaQwenAdmissionRunner("cuda_qwen_admission_fused_invalid")

    with pytest.raises(huggingface_test_runner.CudaAdmissionUnavailable):
        runner.cuda_fused_kernel_mode()


def test_cuda_persistent_baseline_defaults_to_pe16(monkeypatch):
    CudaQwenAdmissionRunner = get_cuda_qwen_admission_runner()

    monkeypatch.delenv("NNCASE_CUDA_REQUIRED_PE", raising=False)
    monkeypatch.delenv("NNCASE_CUDA_TILE_PE", raising=False)
    monkeypatch.delenv("NNCASE_CUDA_FUSED_KERNEL", raising=False)
    monkeypatch.delenv("NNCASE_CUDA_PE_GMEM_LIMIT_BYTES", raising=False)
    runner = CudaQwenAdmissionRunner("cuda_qwen_admission_persistent_pe")

    assert runner.cuda_required_pe_count() == 16
    assert runner.make_cuda_pe_target_options(16).FusedKernelMode == "off"


def test_cuda_pe_gmem_limit_is_written_to_target_options(monkeypatch):
    CudaQwenAdmissionRunner = get_cuda_qwen_admission_runner()

    limit = 64 * 1024 * 1024
    monkeypatch.setenv("NNCASE_CUDA_PE_GMEM_LIMIT_BYTES", str(limit))
    runner = CudaQwenAdmissionRunner("cuda_qwen_admission_gmem_limit")

    options = runner.make_cuda_pe_target_options(16)

    assert list(options.MemoryCapacities) == [524288, limit]


def test_cuda_pe_gmem_limit_rejects_invalid_values(monkeypatch):
    CudaQwenAdmissionRunner = get_cuda_qwen_admission_runner()
    import huggingface_test_runner

    monkeypatch.setenv("NNCASE_CUDA_PE_GMEM_LIMIT_BYTES", "0")
    runner = CudaQwenAdmissionRunner("cuda_qwen_admission_gmem_limit_invalid")

    with pytest.raises(huggingface_test_runner.CudaAdmissionUnavailable):
        runner.make_cuda_pe_target_options(16)


def test_cuda_pe_scheduler_rebuild_uses_chosen_pe_without_touching_text_input(monkeypatch):
    CudaQwenAdmissionRunner = get_cuda_qwen_admission_runner()

    monkeypatch.setenv("NNCASE_CUDA_SM_COUNT", "8")
    runner = CudaQwenAdmissionRunner("cuda_qwen_admission_scheduler")
    runner.num_blocks = 16
    runner.max_model_len = 256
    runner.kv_cache_config = object()
    runner.inputs = [dict(name="input_ids", data=["tokenized"])]
    runner.calibs = [dict(name="input_ids", data=["calib"])]
    calls = []

    class FakeScheduler:
        def __init__(self, config, num_blocks, max_model_len, hierarchy):
            calls.append((config, num_blocks, max_model_len, hierarchy))

    monkeypatch.setattr("huggingface_test_runner.nncase.PagedAttentionScheduler", FakeScheduler)
    monkeypatch.setattr("huggingface_test_runner.nncase._nncase.RefPagedAttentionScheduler", FakeScheduler)

    runner.rebuild_paged_attention_schedulers([4])

    assert runner.inputs[0]["data"] == ["tokenized"]
    assert runner.calibs[0]["data"] == ["calib"]
    assert runner.inputs[1]["scheduler"] is not runner.inputs[2]["scheduler"]
    assert runner.inputs[2]["scheduler"] is not runner.calibs[2]["scheduler"]
    assert all(call == (runner.kv_cache_config, 16, 256, [4]) for call in calls)


def test_cuda_required_pe_expands_num_blocks_for_num_blocks_sharding(monkeypatch):
    CudaQwenAdmissionRunner = get_cuda_qwen_admission_runner()
    import huggingface_test_runner

    monkeypatch.setenv("NNCASE_CUDA_SM_COUNT", "20")
    monkeypatch.setenv("NNCASE_CUDA_REQUIRED_PE", "20")
    runner = CudaQwenAdmissionRunner("cuda_qwen_admission_required_pe_blocks")
    runner.block_size = 256
    runner.num_blocks = 16
    runner.max_sessions = 16
    runner.max_model_len = 256
    runner.sharding_axes = [huggingface_test_runner.nncase.PagedKVCacheDimKind.NumBlocks]
    runner.axis_policies = [[0]]

    adjusted = runner.ensure_cuda_num_blocks_for_pe(20)

    assert adjusted is True
    assert runner.num_blocks == 20
    assert runner.max_model_len == 256
    assert runner.max_model_len % runner.block_size == 0
