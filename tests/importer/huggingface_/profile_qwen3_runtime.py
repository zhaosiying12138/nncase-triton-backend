# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Profile existing Qwen3 CPU/CUDA kmodels without recompiling them."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import nncase
import numpy as np
from ml_dtypes import bfloat16  # noqa: F401 - registers bfloat16 with numpy/pybind dtype conversion.
from transformers import AutoConfig, AutoTokenizer


DEFAULT_PROMPT = "为什么说包上恩主演的《江湖夜雨十年灯》是国产电视剧的巅峰之作？"
DEFAULT_CPU_KMODEL = Path("tests_output/test_qwen3/infer/cpu/noptq/test.kmodel")
DEFAULT_CUDA_KMODEL = Path("tests_output/test_qwen3_cuda_poc/infer/cuda/noptq/test.kmodel")
DEFAULT_PROFILE_DIR = Path("tests_output/qwen3_runtime_profile")
DEFAULT_CUDA_RECOMPILE_TEST = "tests/importer/huggingface_/test_qwen3_cuda.py::test_qwen3_cuda_poc"


@dataclass
class TargetRuntime:
    target: str
    pe_count: int
    kmodel: Path
    input_ids: np.ndarray
    tokenizer: object
    scheduler: object


def _dtype_from_name(name: str):
    if name == "float32":
        return np.float32
    if name == "float16":
        return np.float16
    raise ValueError(f"unsupported paged attention dtype: {name}")


def _find_qwen3_model_dir(explicit: str | None) -> Path:
    if explicit:
        model_dir = Path(explicit).expanduser()
        if not model_dir.exists():
            raise FileNotFoundError(model_dir)
        return model_dir

    local = Path("tests/llm/Qwen/Qwen3-0.6B")
    if (local / "config.json").exists():
        return local

    hf_root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B"
    ref = hf_root / "refs/main"
    if ref.exists():
        snapshot = hf_root / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if (snapshot / "config.json").exists():
            return snapshot

    raise FileNotFoundError("cannot find local Qwen/Qwen3-0.6B model directory")


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    default_prompt_file = Path("tests/importer/huggingface_/prompt_qwen3_profile.txt")
    if default_prompt_file.exists():
        return default_prompt_file.read_text(encoding="utf-8").strip()
    return DEFAULT_PROMPT


def _build_input_ids(tokenizer, prompt: str) -> np.ndarray:
    messages = [
        {"role": "system", "content": "You are a assistant!"},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer([text], return_tensors="np").input_ids[0].astype(np.int64)


def _make_paged_attention_config(model_dir: Path, target: str, pe_count: int, args: argparse.Namespace):
    hf_config = AutoConfig.from_pretrained(str(model_dir / "config.json"))
    head_dim = getattr(
        hf_config,
        "head_dim",
        hf_config.hidden_size // hf_config.num_attention_heads,
    )
    cache_layout = [
        nncase.PagedKVCacheDimKind.NumBlocks,
        nncase.PagedKVCacheDimKind.NumLayers,
        nncase.PagedKVCacheDimKind.KV,
        nncase.PagedKVCacheDimKind.NumKVHeads,
        nncase.PagedKVCacheDimKind.HeadDim,
        nncase.PagedKVCacheDimKind.BlockSize,
    ]
    if target == "cuda":
        vectorized_axes = []
        vector_lanes = []
    else:
        vectorized_axes = [nncase.PagedKVCacheDimKind.HeadDim]
        vector_lanes = [8]
    sharding_axes = [nncase.PagedKVCacheDimKind.NumBlocks]
    axis_policies = [[0]]
    return nncase.PagedAttentionConfig(
        hf_config.num_hidden_layers,
        hf_config.num_key_value_heads,
        head_dim,
        np.dtype(_dtype_from_name("float32")),
        int(args.block_size),
        cache_layout,
        vectorized_axes,
        vector_lanes,
        sharding_axes,
        axis_policies,
    ), int(args.num_blocks), int(args.max_model_len), [int(pe_count)]


def _make_target_runtime(target: str, args: argparse.Namespace, tokenizer, input_ids) -> TargetRuntime:
    pe_count = 1 if target == "cpu" else int(args.cuda_pe)
    kmodel = Path(args.cpu_kmodel if target == "cpu" else args.cuda_kmodel)
    if not kmodel.exists():
        raise FileNotFoundError(
            f"{target} kmodel not found: {kmodel}. Re-run the CPU/CUDA Qwen tests once to populate the kmodel cache."
        )
    config, num_blocks, max_model_len, hierarchy = _make_paged_attention_config(args.model_dir, target, pe_count, args)
    scheduler = nncase.PagedAttentionScheduler(config, num_blocks, max_model_len, hierarchy)
    return TargetRuntime(target, pe_count, kmodel, input_ids.copy(), tokenizer, scheduler)


def _load_sim(kmodel: Path):
    sim = nncase.Simulator()
    begin = time.perf_counter()
    with open(kmodel, "rb") as f:
        sim.load_model(f)
    return sim, time.perf_counter() - begin


def _set_inputs(sim, input_ids: np.ndarray, kv_object):
    for index, value in enumerate([input_ids, kv_object]):
        if isinstance(value, nncase.PagedAttentionKVCache):
            tensor = nncase.RuntimeTensor.from_object(value)
        else:
            tensor = nncase.RuntimeTensor.from_numpy(np.asarray(value))
        sim.set_input_tensor(index, tensor)


def _decode_token(tokenizer, logits: np.ndarray):
    token_id = int(np.argmax(logits[-1, :], axis=-1))
    token = tokenizer.decode(token_id, skip_special_tokens=False)
    return token_id, token


def _make_stream_token_callback(target: str, enabled: bool, stream=None):
    if not enabled:
        return None
    output = stream if stream is not None else sys.stdout
    print(f"\n=== streaming generated text ({target}) ===", file=output, flush=True)

    def emit(token: str):
        print(token, end="", file=output, flush=True)

    return emit


def _compile_refresh_command(target: str):
    if target != "cuda":
        return None
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-s",
        DEFAULT_CUDA_RECOMPILE_TEST,
    ]


def _compile_refresh_cleanup_paths(target: str):
    if target != "cuda":
        return []
    return [DEFAULT_CUDA_KMODEL.parents[3]]


def _refresh_compile_artifacts(target: str, mode: str):
    if mode == "reuse":
        return
    if mode != "always":
        raise ValueError(f"unsupported compile cache mode: {mode}")
    command = _compile_refresh_command(target)
    if command is None:
        return
    for path in _compile_refresh_cleanup_paths(target):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    print(f"[qwen3-profile] refreshing compile artifacts: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def _synchronize(target: str):
    if target != "cuda":
        return
    import torch

    torch.cuda.synchronize()


def _run_tokens(runtime: TargetRuntime, tokens: int, token_callback=None):
    sim, load_seconds = _load_sim(runtime.kmodel)
    input_ids = runtime.input_ids.copy()
    step_seconds: list[float] = []
    token_ids: list[int] = []
    token_text: list[str] = []

    for step in range(tokens):
        current_length = int(input_ids.shape[-1])
        kv_object = runtime.scheduler.schedule([0], [current_length])
        _set_inputs(sim, input_ids, kv_object)
        _synchronize(runtime.target)
        begin = time.perf_counter()
        sim.run()
        _synchronize(runtime.target)
        elapsed = time.perf_counter() - begin

        logits = sim.get_output_tensor(0).to_numpy()
        token_id, token = _decode_token(runtime.tokenizer, logits)
        if token_callback is not None:
            token_callback(token)
        input_ids = np.array([token_id], dtype=np.int64)
        step_seconds.append(elapsed)
        token_ids.append(token_id)
        token_text.append(token)

    return {
        "load_seconds_excluded": load_seconds,
        "step_seconds": step_seconds,
        "token_ids": token_ids,
        "tokens": token_text,
    }


def _summarize(run: dict) -> dict:
    steps = run["step_seconds"]
    decode_steps = steps[1:]
    run["ttft_seconds"] = steps[0] if steps else None
    run["tpot_seconds"] = statistics.mean(decode_steps) if decode_steps else None
    run["decode_median_seconds"] = statistics.median(decode_steps) if decode_steps else None
    run["total_step_seconds"] = sum(steps)
    run["tokens_per_second"] = len(steps) / run["total_step_seconds"] if steps else 0.0
    run["decode_tokens_per_second"] = (
        len(decode_steps) / sum(decode_steps) if decode_steps else None
    )
    return run


def _print_table(results: dict):
    print("\n=== qwen3 runtime profile (sim.run only) ===")
    print("target  pe  prompt_tokens  generated  ttft_s  tpot_s  total_s  tok/s")
    for target, result in results.items():
        print(
            f"{target:<6} {result['pe_count']:<3} {result['prompt_tokens']:<13} "
            f"{len(result['step_seconds']):<9} {result['ttft_seconds']:.3f}  "
            f"{(result['tpot_seconds'] or 0.0):.3f}  {result['total_step_seconds']:.3f}  "
            f"{result['tokens_per_second']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="cpu,cuda", help="comma separated targets: cpu,cuda")
    parser.add_argument("--tokens", "--max-new-tokens", dest="tokens", type=int, default=8, help="number of generated tokens to measure")
    parser.add_argument("--warmup-tokens", type=int, default=1, help="tokens for isolated warmup, excluded")
    parser.add_argument("--prompt", default=None, help="inline prompt; defaults to prompt_qwen3_profile.txt")
    parser.add_argument("--prompt-file", default=None, help="prompt file override")
    parser.add_argument("--model-dir", default=None, help="local Qwen3-0.6B directory")
    parser.add_argument("--cpu-kmodel", default=str(DEFAULT_CPU_KMODEL))
    parser.add_argument("--cuda-kmodel", default=str(DEFAULT_CUDA_KMODEL))
    parser.add_argument("--cuda-pe", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR))
    parser.add_argument("--verbose", "--verboses", "--verbose-triton-launches", action="store_true", help="print Triton launch details")
    parser.add_argument("--verbose-limit", type=int, default=120)
    parser.add_argument("--stream-tokens", action="store_true", help="print each measured generated token immediately")
    parser.add_argument("--compile-cache-mode", choices=("reuse", "always"), default="reuse", help="reuse existing kmodel or always re-run CUDA nncase passes/Triton codegen before profiling")
    parser.add_argument("--triton-cache-dir", default=None, help="cache generated Triton source/bytecode")
    parser.add_argument("--jsonl", default=None, help="append summary records to this jsonl file")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tokens < 1:
        raise ValueError("--tokens must be positive")
    args.model_dir = _find_qwen3_model_dir(args.model_dir)
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    if "cuda" in targets:
        os.environ.setdefault("NNCASE_CUDA_SM_COUNT", str(int(args.cuda_pe)))
        os.environ.setdefault("NNCASE_CUDA_REQUIRED_PE", str(int(args.cuda_pe)))
        os.environ.setdefault("NNCASE_CUDA_USE_NATIVE_TRITON_KERNELS", "1")
        os.environ.setdefault("NNCASE_CUDA_REQUIRE_TRITON_KERNELS", "1")
        os.environ.setdefault("NNCASE_CUDA_FP32_PARTIALS", "1")

    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    triton_cache = Path(args.triton_cache_dir) if args.triton_cache_dir else profile_dir / "triton_cache"
    triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NNCASE_TRITON_CACHE_DIR", str(triton_cache.resolve()))
    if args.verbose:
        os.environ["NNCASE_TRITON_VERBOSE"] = "1"
        os.environ["NNCASE_TRITON_VERBOSE_LIMIT"] = str(max(0, args.verbose_limit))

    prompt = _read_prompt(args)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    input_ids = _build_input_ids(tokenizer, prompt)
    if int(input_ids.shape[-1]) > 1024:
        raise ValueError(f"prefill prompt has {input_ids.shape[-1]} tokens, but the compiled shape bucket supports <= 1024")
    if int(input_ids.shape[-1]) + args.tokens > args.max_model_len:
        raise ValueError(
            f"prompt_tokens + tokens = {int(input_ids.shape[-1]) + args.tokens} exceeds --max-model-len={args.max_model_len}"
        )
    results = {}
    for target in targets:
        if target not in ("cpu", "cuda"):
            raise ValueError(f"unsupported target: {target}")
        if target == "cuda" and not nncase.check_target("cuda"):
            raise RuntimeError("nncase cuda target is not available")

        _refresh_compile_artifacts(target, args.compile_cache_mode)

        if args.warmup_tokens > 0:
            warm = _make_target_runtime(target, args, tokenizer, input_ids)
            _run_tokens(warm, args.warmup_tokens)
            del warm
            gc.collect()

        runtime = _make_target_runtime(target, args, tokenizer, input_ids)
        stream_callback = _make_stream_token_callback(target, args.stream_tokens)
        measured = _summarize(_run_tokens(runtime, args.tokens, stream_callback))
        if stream_callback is not None:
            print("", flush=True)
        measured["target"] = target
        measured["pe_count"] = runtime.pe_count
        measured["kmodel"] = str(runtime.kmodel)
        measured["prompt_tokens"] = int(input_ids.shape[-1])
        measured["prompt"] = prompt
        measured["model_dir"] = str(args.model_dir)
        results[target] = measured
        del runtime
        gc.collect()

    if "cpu" in results and "cuda" in results:
        results["comparison"] = {
            "cuda_ttft_speedup": results["cpu"]["ttft_seconds"] / results["cuda"]["ttft_seconds"],
            "cuda_tpot_speedup": (
                results["cpu"]["tpot_seconds"] / results["cuda"]["tpot_seconds"]
                if results["cpu"]["tpot_seconds"] and results["cuda"]["tpot_seconds"]
                else None
            ),
            "cuda_total_speedup": results["cpu"]["total_step_seconds"] / results["cuda"]["total_step_seconds"],
        }

    _print_table({k: v for k, v in results.items() if k in ("cpu", "cuda")})
    print("\n=== generated text ===")
    for target in [k for k in targets if k in results]:
        print(f"{target}: {''.join(results[target]['tokens'])}")

    output_path = profile_dir / f"qwen3_runtime_profile_{int(time.time())}.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as f:
            for target in [k for k in targets if k in results]:
                f.write(json.dumps(results[target], ensure_ascii=False) + "\n")
    print(f"\nprofile_json={output_path}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
