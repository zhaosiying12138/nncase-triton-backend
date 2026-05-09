import pytest


def get_profile_module(monkeypatch):
    import importlib.util
    import sys
    import types
    from pathlib import Path

    monkeypatch.setitem(sys.modules, "nncase", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ml_dtypes", types.SimpleNamespace(bfloat16=object()))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoConfig=object(), AutoTokenizer=object()),
    )
    try:
        module_name = "_profile_qwen3_runtime_under_test"
        module_path = (
            Path(__file__).resolve().parents[1]
            / "importer"
            / "huggingface_"
            / "profile_qwen3_runtime.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        profile_qwen3_runtime = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, profile_qwen3_runtime)
        spec.loader.exec_module(profile_qwen3_runtime)
    except (ModuleNotFoundError, RuntimeError) as ex:
        pytest.skip(f"qwen3 profile module is not importable: {ex}")
    return profile_qwen3_runtime


class RecordingStream:
    def __init__(self):
        self.parts = []
        self.flush_count = 0

    def write(self, value):
        self.parts.append(value)

    def flush(self):
        self.flush_count += 1


def test_stream_token_callback_prints_each_token_immediately(monkeypatch):
    profile = get_profile_module(monkeypatch)
    stream = RecordingStream()

    callback = profile._make_stream_token_callback("cuda", True, stream=stream)

    assert callback is not None
    callback("<think>")
    callback("\n")
    callback("好的")

    assert "".join(stream.parts) == "\n=== streaming generated text (cuda) ===\n<think>\n好的"
    assert stream.flush_count >= 4


def test_stream_token_callback_can_be_disabled(monkeypatch):
    profile = get_profile_module(monkeypatch)

    assert profile._make_stream_token_callback("cuda", False) is None


def test_compile_refresh_command_defaults_to_cuda_pytest(monkeypatch):
    profile = get_profile_module(monkeypatch)

    command = profile._compile_refresh_command("cuda")

    assert command[:3] == [profile.sys.executable, "-m", "pytest"]
    assert "tests/importer/huggingface_/test_qwen3_cuda.py::test_qwen3_cuda_poc" in command


def test_compile_refresh_is_cuda_only(monkeypatch):
    profile = get_profile_module(monkeypatch)

    assert profile._compile_refresh_command("cpu") is None


def test_paged_attention_config_matches_cuda_compile_layout(monkeypatch, tmp_path):
    profile = get_profile_module(monkeypatch)
    kind = type(
        "Kind",
        (),
        {
            "NumBlocks": "NumBlocks",
            "NumLayers": "NumLayers",
            "KV": "KV",
            "NumKVHeads": "NumKVHeads",
            "HeadDim": "HeadDim",
            "BlockSize": "BlockSize",
        },
    )
    captured = {}

    class FakeConfig:
        num_hidden_layers = 28
        num_key_value_heads = 8
        hidden_size = 1024
        num_attention_heads = 16
        head_dim = 128

    def fake_paged_config(*args):
        captured["args"] = args
        return "config"

    monkeypatch.setattr(profile, "AutoConfig", type("AutoConfig", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: FakeConfig())}))
    monkeypatch.setattr(profile.nncase, "PagedKVCacheDimKind", kind, raising=False)
    monkeypatch.setattr(profile.nncase, "PagedAttentionConfig", fake_paged_config, raising=False)
    args = type("Args", (), {"block_size": 256, "num_blocks": 16, "max_model_len": 4096})()

    config, _, _, _ = profile._make_paged_attention_config(tmp_path, "cuda", 16, args)

    assert config == "config"
    assert captured["args"][6] == []
    assert captured["args"][7] == []


def test_cuda_profile_defaults_to_strict_triton(monkeypatch, tmp_path):
    profile = get_profile_module(monkeypatch)
    for name in (
        "NNCASE_CUDA_SM_COUNT",
        "NNCASE_CUDA_REQUIRED_PE",
        "NNCASE_CUDA_USE_NATIVE_TRITON_KERNELS",
        "NNCASE_CUDA_REQUIRE_TRITON_KERNELS",
        "NNCASE_CUDA_FP32_PARTIALS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(profile, "_find_qwen3_model_dir", lambda value: tmp_path)
    monkeypatch.setattr(profile, "_read_prompt", lambda args: "hello")
    monkeypatch.setattr(profile, "AutoTokenizer", type("AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: object())}))
    monkeypatch.setattr(profile, "_build_input_ids", lambda tokenizer, prompt: type("Ids", (), {"shape": [1], "copy": lambda self: self})())
    monkeypatch.setattr(profile.nncase, "check_target", lambda target: True, raising=False)
    monkeypatch.setattr(profile, "_refresh_compile_artifacts", lambda target, mode: None)
    monkeypatch.setattr(profile, "_make_target_runtime", lambda *args, **kwargs: type("Runtime", (), {"pe_count": 16, "kmodel": tmp_path / "test.kmodel"})())
    monkeypatch.setattr(profile, "_run_tokens", lambda *args, **kwargs: {"step_seconds": [1.0], "token_ids": [1], "tokens": ["x"], "load_seconds_excluded": 0.0})
    monkeypatch.setattr(profile, "_print_table", lambda results: None)
    monkeypatch.setattr(profile.sys, "argv", [
        "profile_qwen3_runtime.py",
        "--targets",
        "cuda",
        "--tokens",
        "1",
        "--warmup-tokens",
        "0",
        "--profile-dir",
        str(tmp_path / "profile"),
    ])

    profile.main()

    assert profile.os.environ["NNCASE_CUDA_SM_COUNT"] == "16"
    assert profile.os.environ["NNCASE_CUDA_REQUIRED_PE"] == "16"
    assert profile.os.environ["NNCASE_CUDA_USE_NATIVE_TRITON_KERNELS"] == "1"
    assert profile.os.environ["NNCASE_CUDA_REQUIRE_TRITON_KERNELS"] == "1"
    assert profile.os.environ["NNCASE_CUDA_FP32_PARTIALS"] == "1"
