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
