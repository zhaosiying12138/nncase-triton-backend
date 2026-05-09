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
