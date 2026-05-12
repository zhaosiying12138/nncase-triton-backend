#!/usr/bin/env python3
"""Generate Qwen3 PE=16 layer0 graphs for fused-kernel mode comparisons."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
BASE_GENERATOR = ROOT / "docs/triton-backend/generate_qwen3_compute_layer0.py"
CONFIG_PATH = ROOT / "tests/llm/Qwen/Qwen3-0.6B/config.json"
HF_QWEN3_SOURCE_URL = (
    "https://github.com/huggingface/transformers/blob/main/"
    "src/transformers/models/qwen3/modeling_qwen3.py"
)
COMPARE_STEM = "qwen3-off-vs-compute-ccl-layer0"
VLLM_COMPARE_STEM = "qwen3-vllm-tp16-vs-nncase-off-layer0"


def load_base_module():
    spec = importlib.util.spec_from_file_location("qwen3_compute_layer0_base", BASE_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base generator: {BASE_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    title_mode: str
    stem: str
    meta_path: Path
    launch_summary_path: Path
    triton_module_path: Path
    function_name: str
    ordinals: tuple[int, ...]
    layer0_end: int
    boundary: int | None
    layout_rows: tuple
    nn_data_edges: tuple[tuple[int, int], ...]
    nn_skip_edges: frozenset[tuple[int, int]]
    hf_to_nn_mapping: tuple[tuple[str, int], ...]
    arg_roles: dict[int, dict[str, str]]
    io_keys: dict[int, dict[str, tuple[str, ...]]]
    hf_meaning: dict[int, str]
    helpers: dict[int, str]
    required_fusions: tuple[str, ...] = ()

    @property
    def out_dot(self) -> Path:
        return OUT_DIR / f"{self.stem}.dot"

    @property
    def out_svg(self) -> Path:
        return OUT_DIR / f"{self.stem}.svg"

    @property
    def out_md(self) -> Path:
        return OUT_DIR / f"{self.stem}.md"


def p(path: str) -> Path:
    return ROOT / path


COMPUTE_CCL_SPEC = ModeSpec(
    mode="compute-ccl",
    title_mode="compute-ccl",
    stem="qwen3-compute-ccl-layer0",
    meta_path=p("tests_output/test_qwen3_cuda_poc/cuda_admission/pe_16/CodeGen/cuda/cuda_meta.json"),
    launch_summary_path=p("tests_output/test_qwen3_cuda_poc/cuda_admission/pe_16/CodeGen/cuda/launch_summary.txt"),
    triton_module_path=p("tests_output/test_qwen3_cuda_poc/cuda_admission/pe_16/CodeGen/cuda/triton_module.py"),
    function_name="main_segment_0_prim",
    ordinals=tuple(range(31)),
    layer0_end=29,
    boundary=30,
    layout_rows=base.LAYOUT_ROWS,
    nn_data_edges=base.NN_DATA_EDGES,
    nn_skip_edges=base.NN_SKIP_EDGES,
    hf_to_nn_mapping=base.HF_TO_NN_MAPPING,
    arg_roles=base.ARG_ROLES,
    io_keys=base.IO_KEYS,
    hf_meaning=base.HF_MEANING,
    helpers=base.HELPERS,
    required_fusions=(
        "fusion.cuda.layer_norm_matmul",
        "fusion.cuda.layer_norm_transpose",
        "fusion.cuda.mul_cos",
        "fusion.cuda.mul_sin",
        "fusion.cuda.rope",
        "fusion.cuda.matmul_silu_matmul_mul_matmul",
    ),
)


OFF_LAYOUT_ROWS = (
    base.Row(("hf_input",), ("n_00",)),
    base.Row((), ("n_01",)),
    base.Row(("hf_embed",), ("n_02",)),
    base.Row((), ("n_03",)),
    base.Row((), ("n_04",)),
    base.Row(("hf_input_ln",), ("n_05",)),
    base.Row(("hf_q_proj",), ()),
    base.Row(("hf_q_norm",), ("n_06",)),
    base.Row(("hf_rotary",), ("n_07", "n_08")),
    base.Row((), ("n_09",)),
    base.Row(("hf_rope_q",), ("n_10", "n_11")),
    base.Row(("hf_v_proj",), ("n_12", "n_13")),
    base.Row((), ("n_14",)),
    base.Row(("hf_k_proj",), ("n_15",)),
    base.Row(("hf_k_norm",), ("n_16",)),
    base.Row(("hf_rope_k",), ("n_17", "n_18")),
    base.Row(("hf_cache",), ("n_19", "n_20")),
    base.Row(("hf_attn",), ("n_21", "n_22")),
    base.Row(("hf_o_proj",), ("n_23",)),
    base.Row(("hf_attn_resid",), ("n_24",)),
    base.Row(("hf_post_ln",), ()),
    base.Row(("hf_gate",), ()),
    base.Row(("hf_up",), ("n_25",)),
    base.Row(("hf_act_mul",), ("n_26",)),
    base.Row(("hf_down",), ("n_27",)),
    base.Row(("hf_mlp_resid",), ("n_28",)),
    base.Row(("hf_layer1",), ("n_29",)),
)


OFF_NN_DATA_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 10),
    (7, 8),
    (8, 9),
    (8, 10),
    (9, 10),
    (10, 11),
    (11, 21),
    (5, 12),
    (12, 13),
    (13, 14),
    (14, 20),
    (12, 15),
    (15, 16),
    (16, 17),
    (8, 17),
    (9, 17),
    (17, 18),
    (18, 19),
    (19, 21),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (4, 24),
    (24, 25),
    (24, 26),
    (25, 26),
    (26, 27),
    (27, 28),
    (24, 28),
    (28, 29),
)


OFF_HF_TO_NN_MAPPING = (
    ("hf_input", 0),
    ("hf_input", 1),
    ("hf_embed", 2),
    ("hf_embed", 3),
    ("hf_embed", 4),
    ("hf_input_ln", 5),
    ("hf_q_proj", 5),
    ("hf_q_norm", 6),
    ("hf_rotary", 7),
    ("hf_rotary", 8),
    ("hf_rotary", 9),
    ("hf_rope_q", 10),
    ("hf_rope_q", 11),
    ("hf_input_ln", 12),
    ("hf_v_proj", 13),
    ("hf_v_proj", 14),
    ("hf_k_proj", 15),
    ("hf_k_norm", 16),
    ("hf_rope_k", 17),
    ("hf_rope_k", 18),
    ("hf_cache", 19),
    ("hf_cache", 20),
    ("hf_attn", 21),
    ("hf_attn", 22),
    ("hf_o_proj", 23),
    ("hf_attn_resid", 24),
    ("hf_post_ln", 24),
    ("hf_gate", 24),
    ("hf_up", 25),
    ("hf_act_mul", 26),
    ("hf_down", 27),
    ("hf_mlp_resid", 28),
    ("hf_layer1", 28),
    ("hf_layer1", 29),
)


OFF_ARG_ROLES = {
    0: {"arg1": "input_ids", "arg0": "ids_local"},
    1: {"arg0": "input_ids", "arg1": "pad_id", "arg3": "mask"},
    2: {"arg0": "embed_w", "arg1": "input_ids", "arg2": "embed"},
    3: {"arg0": "mask", "arg1": "fill_value", "arg2": "embed", "arg4": "masked_embed"},
    4: {"arg0": "embed_shard", "arg1": "hidden"},
    5: {"arg0": "q_w", "arg1": "hidden", "arg2": "rms_w", "arg3": "rms_bias", "arg5": "q_proj", "arg6": "normed_hidden"},
    6: {"arg0": "q_view", "arg1": "q_norm_w", "arg2": "q_norm_bias", "arg4": "q_norm_t"},
    7: {"arg0": "kvCache", "arg2": "position_ids"},
    8: {"arg0": "position_ids", "arg1": "inv_freq", "arg3": "cos", "arg4": "angle"},
    9: {"arg0": "angle", "arg2": "sin"},
    10: {"arg0": "q", "arg1": "cos", "arg2": "sin", "arg3": "q_rope"},
    11: {"arg0": "q_rope", "arg2": "q_attn"},
    12: {"arg0": "normed_hidden", "arg1": "kv_input"},
    13: {"arg0": "kv_input", "arg1": "v_w", "arg2": "v_partial"},
    14: {"arg0": "v_view", "arg2": "v_cache"},
    15: {"arg0": "kv_input", "arg1": "k_w", "arg2": "k_partial"},
    16: {"arg0": "k_view", "arg1": "k_norm_w", "arg2": "k_norm_bias", "arg4": "k_norm_t"},
    17: {"arg0": "k", "arg1": "cos", "arg2": "sin", "arg3": "k_rope"},
    18: {"arg0": "k_rope", "arg2": "k_cache"},
    19: {"arg0": "k_cache", "arg1": "kvCache"},
    20: {"arg0": "v_cache", "arg1": "kvCache"},
    21: {"arg0": "q_attn", "arg1": "kvCache", "arg2": "workspace", "arg3": "scale", "arg4": "attn_out"},
    22: {"arg0": "attn_out", "arg2": "o_proj_in"},
    23: {"arg0": "o_proj_in", "arg1": "o_w", "arg2": "o_partial"},
    24: {
        "arg0": "gate_w",
        "arg1": "post_norm_w",
        "arg2": "post_norm_bias",
        "arg3": "residual0",
        "arg4": "o_partial",
        "arg6": "gate",
        "arg7": "residual1",
        "arg8": "post_norm",
    },
    25: {"arg0": "post_norm", "arg1": "up_w", "arg3": "up"},
    26: {"arg0": "up", "arg1": "gate", "arg3": "mlp_hidden"},
    27: {"arg0": "mlp_hidden", "arg1": "down_w", "arg2": "mlp_partial"},
    28: {
        "arg0": "layer1_q_w",
        "arg1": "layer1_norm_w",
        "arg2": "layer1_norm_bias",
        "arg3": "residual1",
        "arg4": "mlp_partial",
        "arg6": "layer1_q",
        "arg7": "layer1_pre_norm",
        "arg8": "layer0_out",
    },
    29: {"arg0": "layer1_q_view", "arg1": "layer1_q_norm_w", "arg2": "layer1_q_norm_bias", "arg4": "layer1_q_norm_t"},
}


OFF_IO_KEYS = {
    0: {"in": ("arg1",), "out": ("arg0",)},
    1: {"in": ("arg0",), "param": ("arg1",), "out": ("arg3",)},
    2: {"in": ("arg1",), "param": ("arg0",), "out": ("arg2",)},
    3: {"in": ("arg0", "arg2"), "param": ("arg1",), "out": ("arg4",)},
    4: {"in": ("arg0",), "out": ("arg1",)},
    5: {"in": ("arg1",), "param": ("arg0", "arg2", "arg3"), "out": ("arg5", "arg6")},
    6: {"in": ("arg0",), "param": ("arg1", "arg2"), "out": ("arg4",)},
    7: {"in": ("arg0",), "out": ("arg2",)},
    8: {"in": ("arg0",), "param": ("arg1",), "out": ("arg3", "arg4")},
    9: {"in": ("arg0",), "out": ("arg2",)},
    10: {"in": ("arg0", "arg1", "arg2"), "out": ("arg3",)},
    11: {"in": ("arg0",), "out": ("arg2",)},
    12: {"in": ("arg0",), "out": ("arg1",)},
    13: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    14: {"in": ("arg0",), "out": ("arg2",)},
    15: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    16: {"in": ("arg0",), "param": ("arg1", "arg2"), "out": ("arg4",)},
    17: {"in": ("arg0", "arg1", "arg2"), "out": ("arg3",)},
    18: {"in": ("arg0",), "out": ("arg2",)},
    19: {"in": ("arg0", "arg1"), "out": ("arg1",)},
    20: {"in": ("arg0", "arg1"), "out": ("arg1",)},
    21: {"in": ("arg0", "arg1"), "param": ("arg2", "arg3"), "out": ("arg4",)},
    22: {"in": ("arg0",), "out": ("arg2",)},
    23: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    24: {"in": ("arg3", "arg4"), "param": ("arg0", "arg1", "arg2"), "out": ("arg6", "arg7", "arg8")},
    25: {"in": ("arg0",), "param": ("arg1",), "out": ("arg3",)},
    26: {"in": ("arg0", "arg1"), "out": ("arg3",)},
    27: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    28: {"in": ("arg3", "arg4"), "param": ("arg0", "arg1", "arg2"), "out": ("arg6", "arg7", "arg8")},
    29: {"in": ("arg0",), "param": ("arg1", "arg2"), "out": ("arg4",)},
}


OFF_HF_MEANING = {
    0: "Qwen3Model.forward input_ids",
    1: "Qwen3Model.forward input_ids mask",
    2: "Qwen3Model.embed_tokens",
    3: "Qwen3Model.embed_tokens materialization",
    4: "Qwen3Model.embed_tokens materialization",
    5: "Qwen3DecoderLayer.input_layernorm + Qwen3Attention.q_proj",
    6: "Qwen3Attention.q_norm + view/transpose",
    7: "Qwen3RotaryEmbedding.forward position_ids",
    8: "Qwen3RotaryEmbedding.forward cos/angle",
    9: "Qwen3RotaryEmbedding.forward sin",
    10: "apply_rotary_pos_emb(q)",
    11: "Q layout for eager_attention_forward",
    12: "K/V normalized hidden materialization",
    13: "Qwen3Attention.v_proj",
    14: "V cache layout",
    15: "Qwen3Attention.k_proj",
    16: "Qwen3Attention.k_norm + view/transpose",
    17: "apply_rotary_pos_emb(k)",
    18: "K cache layout",
    19: "Cache.update(k)",
    20: "Cache.update(v)",
    21: "eager_attention_forward / paged attention",
    22: "attention output layout",
    23: "Qwen3Attention.o_proj",
    24: "attention residual add + post_attention_layernorm + gate_proj",
    25: "Qwen3MLP.up_proj",
    26: "Qwen3MLP.act_fn + mul",
    27: "Qwen3MLP.down_proj",
    28: "MLP residual add; layer0 output; layer1 q side output",
    29: "layer1 boundary: Qwen3Attention.q_norm",
}


OFF_HELPERS = {
    0: "_nncase_ccl_rank4_kernel",
    1: "_nncase_pe_binary_rank4_kernel",
    2: "_nncase_pe_gather_axis0_rank2_kernel",
    3: "_nncase_pe_where_rank4_kernel",
    4: "_nncase_ccl_rank4_kernel",
    5: "_nncase_pe_layer_norm_matmul_kernel + _nncase_pe_layer_norm_kernel",
    6: "_nncase_pe_layer_norm_kernel + _nncase_pe_transpose_rank4_kernel",
    7: "_nncase_pe_position_ids_kernel",
    8: "_nncase_pe_mul_unary_rank4_kernel",
    9: "_nncase_pe_unary_rank4_kernel",
    10: "_nncase_pe_rope_rank4_kernel",
    11: "_nncase_pe_transpose_rank4_kernel",
    12: "_nncase_ccl_rank4_kernel",
    13: "_nncase_pe_matmul_kernel",
    14: "_nncase_pe_transpose_rank4_kernel",
    15: "_nncase_pe_matmul_kernel",
    16: "_nncase_pe_layer_norm_kernel + _nncase_pe_transpose_rank4_kernel",
    17: "_nncase_pe_rope_rank4_kernel",
    18: "_nncase_pe_transpose_rank4_kernel",
    19: "_nncase_pe_update_kv_rank4_kernel",
    20: "_nncase_pe_update_kv_rank4_kernel",
    21: "_nncase_paged_flash_attention_rank3_kernel",
    22: "_nncase_pe_transpose_rank4_kernel",
    23: "_nncase_pe_matmul_kernel",
    24: "_nncase_pe_binary_rank4_kernel + _nncase_pe_layer_norm_matmul_kernel",
    25: "_nncase_pe_matmul_kernel",
    26: "_nncase_pe_swish_mul_rank4_kernel",
    27: "_nncase_pe_matmul_kernel",
    28: "_nncase_pe_binary_rank4_kernel + _nncase_pe_layer_norm_matmul_kernel",
    29: "_nncase_pe_layer_norm_kernel + _nncase_pe_transpose_rank4_kernel",
}


OFF_SPEC = ModeSpec(
    mode="off",
    title_mode="off",
    stem="qwen3-off-layer0",
    meta_path=p("tests_output/qwen3_reshard_off_probe2/cuda_admission/pe_16/CodeGen/cuda/cuda_meta.json"),
    launch_summary_path=p("tests_output/qwen3_reshard_off_probe2/cuda_admission/pe_16/CodeGen/cuda/launch_summary.txt"),
    triton_module_path=p("tests_output/qwen3_reshard_off_probe2/cuda_admission/pe_16/CodeGen/cuda/triton_module.py"),
    function_name="main_segment_0_prim",
    ordinals=tuple(range(30)),
    layer0_end=28,
    boundary=29,
    layout_rows=OFF_LAYOUT_ROWS,
    nn_data_edges=OFF_NN_DATA_EDGES,
    nn_skip_edges=frozenset(((4, 24), (24, 28))),
    hf_to_nn_mapping=OFF_HF_TO_NN_MAPPING,
    arg_roles=OFF_ARG_ROLES,
    io_keys=OFF_IO_KEYS,
    hf_meaning=OFF_HF_MEANING,
    helpers=OFF_HELPERS,
)


SPECS = {
    "off": OFF_SPEC,
    "compute-ccl": COMPUTE_CCL_SPEC,
}


# The user-facing comparison in this directory intentionally uses
# main_segment_1_prim, because that is the local record where ordinal 04 is the
# all-to-all reshard `SBP=(B,S(0)) -> SBP=(S(0),B)`.
MAIN1_OFF_LAYOUT_ROWS = (
    base.Row(("hf_input",), ("n_00", "n_02")),
    base.Row((), ("n_01",)),
    base.Row(("hf_embed",), ("n_03",)),
    base.Row((), ("n_04",)),
    base.Row(("hf_input_ln",), ("n_05", "n_06")),
    base.Row(("hf_q_proj",), ("n_07",)),
    base.Row(("hf_q_norm",), ("n_08",)),
    base.Row(("hf_rotary",), ("n_09", "n_10")),
    base.Row((), ("n_11", "n_12")),
    base.Row(("hf_rope_q",), ("n_13", "n_14")),
    base.Row((), ("n_15",)),
    base.Row(("hf_v_proj",), ("n_16", "n_17")),
    base.Row((), ("n_18",)),
    base.Row(("hf_k_proj", "hf_k_norm"), ("n_19",)),
    base.Row(("hf_rope_k",), ("n_20", "n_21")),
    base.Row(("hf_cache",), ("n_22", "n_23")),
    base.Row(("hf_attn",), ("n_24", "n_25")),
    base.Row(("hf_o_proj",), ("n_26", "n_27")),
    base.Row(("hf_attn_resid",), ("n_28", "n_29")),
    base.Row(("hf_post_ln",), ("n_30",)),
    base.Row(("hf_gate", "hf_up"), ("n_31",)),
    base.Row(("hf_act_mul", "hf_down"), ("n_32",)),
    base.Row(("hf_mlp_resid",), ("n_33",)),
    base.Row(("hf_layer1",), ("n_34",)),
)


MAIN1_COMPUTE_CCL_LAYOUT_ROWS = (
    base.Row(("hf_input",), ("n_00", "n_02")),
    base.Row((), ("n_01",)),
    base.Row(("hf_embed",), ("n_03",)),
    base.Row((), ("n_04",)),
    base.Row(("hf_input_ln",), ("n_05", "n_06")),
    base.Row(("hf_q_proj",), ("n_07",)),
    base.Row(("hf_q_norm",), ("n_08", "n_09")),
    base.Row(("hf_rotary",), ("n_10", "n_11")),
    base.Row((), ("n_12", "n_13")),
    base.Row(("hf_rope_q",), ("n_14", "n_15")),
    base.Row((), ("n_16",)),
    base.Row(("hf_v_proj",), ("n_17", "n_18")),
    base.Row((), ("n_19",)),
    base.Row(("hf_k_proj",), ("n_20",)),
    base.Row(("hf_k_norm",), ("n_21", "n_22")),
    base.Row(("hf_rope_k",), ("n_23", "n_24")),
    base.Row(("hf_cache",), ("n_25", "n_26")),
    base.Row(("hf_attn",), ("n_27", "n_28")),
    base.Row(("hf_o_proj",), ("n_29", "n_30")),
    base.Row(("hf_attn_resid",), ("n_31", "n_32")),
    base.Row(("hf_post_ln",), ("n_33",)),
    base.Row(("hf_gate", "hf_up"), ("n_34",)),
    base.Row(("hf_act_mul", "hf_down"), ()),
    base.Row(("hf_mlp_resid",), ("n_35",)),
    base.Row(("hf_layer1",), ("n_36",)),
)


MAIN1_OFF_NN_DATA_EDGES = (
    (0, 1),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 14),
    (9, 10),
    (10, 11),
    (10, 12),
    (12, 13),
    (11, 14),
    (13, 14),
    (14, 15),
    (15, 24),
    (5, 16),
    (16, 17),
    (17, 18),
    (18, 23),
    (5, 19),
    (19, 20),
    (10, 20),
    (12, 20),
    (20, 21),
    (21, 22),
    (22, 24),
    (23, 24),
    (24, 25),
    (25, 26),
    (26, 27),
    (27, 28),
    (5, 28),
    (28, 29),
    (28, 30),
    (30, 31),
    (31, 32),
    (32, 33),
    (29, 33),
    (33, 34),
)


MAIN1_COMPUTE_CCL_NN_DATA_EDGES = (
    (0, 1),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 15),
    (10, 11),
    (10, 13),
    (11, 12),
    (13, 14),
    (12, 15),
    (14, 15),
    (15, 16),
    (16, 27),
    (5, 17),
    (17, 18),
    (18, 19),
    (19, 26),
    (5, 20),
    (20, 21),
    (21, 22),
    (11, 23),
    (13, 23),
    (22, 23),
    (23, 24),
    (24, 25),
    (25, 27),
    (26, 27),
    (27, 28),
    (28, 29),
    (29, 30),
    (30, 31),
    (5, 31),
    (31, 32),
    (31, 33),
    (33, 34),
    (34, 35),
    (32, 35),
    (35, 36),
)


MAIN1_OFF_HF_TO_NN_MAPPING = (
    ("hf_input", 0),
    ("hf_input", 1),
    ("hf_input", 2),
    ("hf_embed", 3),
    ("hf_embed", 4),
    ("hf_embed", 5),
    ("hf_input_ln", 5),
    ("hf_input_ln", 6),
    ("hf_q_proj", 7),
    ("hf_q_norm", 8),
    ("hf_rotary", 9),
    ("hf_rotary", 10),
    ("hf_rotary", 11),
    ("hf_rotary", 12),
    ("hf_rotary", 13),
    ("hf_rope_q", 14),
    ("hf_rope_q", 15),
    ("hf_v_proj", 16),
    ("hf_v_proj", 17),
    ("hf_v_proj", 18),
    ("hf_k_proj", 5),
    ("hf_k_norm", 19),
    ("hf_rope_k", 20),
    ("hf_rope_k", 21),
    ("hf_cache", 22),
    ("hf_cache", 23),
    ("hf_attn", 24),
    ("hf_attn", 25),
    ("hf_o_proj", 26),
    ("hf_o_proj", 27),
    ("hf_attn_resid", 28),
    ("hf_attn_resid", 29),
    ("hf_post_ln", 28),
    ("hf_post_ln", 30),
    ("hf_gate", 31),
    ("hf_up", 31),
    ("hf_act_mul", 31),
    ("hf_down", 32),
    ("hf_mlp_resid", 33),
    ("hf_layer1", 33),
    ("hf_layer1", 34),
)


MAIN1_COMPUTE_CCL_HF_TO_NN_MAPPING = (
    ("hf_input", 0),
    ("hf_input", 1),
    ("hf_input", 2),
    ("hf_embed", 3),
    ("hf_embed", 4),
    ("hf_embed", 5),
    ("hf_input_ln", 5),
    ("hf_input_ln", 6),
    ("hf_q_proj", 7),
    ("hf_q_norm", 8),
    ("hf_q_norm", 9),
    ("hf_rotary", 10),
    ("hf_rotary", 11),
    ("hf_rotary", 12),
    ("hf_rotary", 13),
    ("hf_rotary", 14),
    ("hf_rope_q", 15),
    ("hf_rope_q", 16),
    ("hf_v_proj", 17),
    ("hf_v_proj", 18),
    ("hf_v_proj", 19),
    ("hf_input_ln", 20),
    ("hf_k_proj", 20),
    ("hf_k_norm", 21),
    ("hf_k_norm", 22),
    ("hf_rope_k", 23),
    ("hf_rope_k", 24),
    ("hf_cache", 25),
    ("hf_cache", 26),
    ("hf_attn", 27),
    ("hf_attn", 28),
    ("hf_o_proj", 29),
    ("hf_o_proj", 30),
    ("hf_attn_resid", 31),
    ("hf_attn_resid", 32),
    ("hf_post_ln", 31),
    ("hf_post_ln", 33),
    ("hf_gate", 34),
    ("hf_up", 34),
    ("hf_act_mul", 34),
    ("hf_down", 34),
    ("hf_mlp_resid", 35),
    ("hf_layer1", 35),
    ("hf_layer1", 36),
)


MAIN1_OFF_ARG_ROLES = {
    0: {"arg1": "input_ids", "arg0": "ids_s"},
    1: {"arg0": "ids_s", "arg1": "pad_id", "arg3": "mask_s"},
    2: {"arg1": "input_ids", "arg0": "ids_b"},
    3: {"arg0": "embed_w", "arg1": "ids_b", "arg2": "embed_b_s"},
    4: {"arg0": "embed_b_s", "arg1": "embed_s_b"},
    5: {"arg0": "k_w", "arg1": "rms_w", "arg2": "rms_bias", "arg3": "mask_s", "arg4": "fill_value", "arg5": "embed_s_b", "arg7": "k_proj", "arg8": "normed_hidden", "arg9": "residual0"},
    6: {"arg0": "normed_hidden", "arg1": "normed_hidden_b"},
    7: {"arg0": "normed_hidden_b", "arg1": "q_w", "arg3": "q_proj"},
    8: {"arg0": "q_view", "arg1": "q_norm_w", "arg2": "q_norm_bias", "arg4": "q_norm_t"},
    9: {"arg0": "kvCache", "arg2": "position_ids"},
    10: {"arg0": "position_ids", "arg1": "inv_freq", "arg3": "cos_s", "arg4": "angle_s"},
    11: {"arg0": "cos_s", "arg1": "cos_b"},
    12: {"arg0": "angle_s", "arg2": "sin_s"},
    13: {"arg0": "sin_s", "arg1": "sin_b"},
    14: {"arg0": "q", "arg1": "cos_b", "arg2": "sin_b", "arg3": "q_rope"},
    15: {"arg0": "q_rope", "arg2": "q_attn"},
    16: {"arg0": "normed_hidden", "arg1": "kv_input"},
    17: {"arg0": "kv_input", "arg1": "v_w", "arg2": "v_partial"},
    18: {"arg0": "v_view", "arg2": "v_cache"},
    19: {"arg0": "k_view", "arg1": "k_norm_w", "arg2": "k_norm_bias", "arg4": "k_norm_t"},
    20: {"arg0": "k", "arg1": "cos_s", "arg2": "sin_s", "arg3": "k_rope"},
    21: {"arg0": "k_rope", "arg2": "k_cache"},
    22: {"arg0": "k_cache", "arg1": "kvCache"},
    23: {"arg0": "v_cache", "arg1": "kvCache"},
    24: {"arg0": "q_attn", "arg1": "kvCache", "arg2": "workspace", "arg3": "scale", "arg4": "attn_out"},
    25: {"arg0": "attn_out", "arg2": "o_proj_in"},
    26: {"arg0": "o_proj_in", "arg1": "o_w", "arg2": "o_partial"},
    27: {"arg0": "o_partial", "arg1": "o_s_b"},
    28: {"arg0": "post_norm_w", "arg1": "post_norm_bias", "arg2": "residual0", "arg3": "o_s_b", "arg5": "post_norm_s", "arg6": "residual1_s"},
    29: {"arg0": "residual1_s", "arg1": "residual1"},
    30: {"arg0": "post_norm_s", "arg1": "post_norm"},
    31: {"arg0": "post_norm", "arg1": "gate_w", "arg2": "post_norm", "arg3": "up_w", "arg5": "mlp_hidden"},
    32: {"arg0": "mlp_hidden", "arg1": "down_w", "arg2": "mlp_partial"},
    33: {"arg0": "layer1_q_w", "arg1": "layer1_norm_w", "arg2": "layer1_norm_bias", "arg3": "residual1", "arg4": "mlp_partial", "arg6": "layer1_q", "arg7": "layer1_pre_norm", "arg8": "layer0_out"},
    34: {"arg0": "layer1_q_view", "arg1": "layer1_q_norm_w", "arg2": "layer1_q_norm_bias", "arg4": "layer1_q_norm_t"},
}


MAIN1_COMPUTE_CCL_ARG_ROLES = {
    0: {"arg1": "input_ids", "arg0": "ids_s"},
    1: {"arg0": "ids_s", "arg1": "pad_id", "arg3": "mask_s"},
    2: {"arg1": "input_ids", "arg0": "ids_b"},
    3: {"arg0": "embed_w", "arg1": "ids_b", "arg2": "embed_b_s"},
    4: {"arg0": "embed_b_s", "arg1": "embed_s_b"},
    5: {"arg0": "rms_w", "arg1": "rms_bias", "arg2": "mask_s", "arg3": "fill_value", "arg4": "embed_s_b", "arg6": "normed_hidden", "arg7": "residual0"},
    6: {"arg0": "normed_hidden", "arg1": "normed_hidden_b"},
    7: {"arg0": "normed_hidden_b", "arg1": "q_w", "arg3": "q_proj"},
    8: {"arg0": "q_view", "arg1": "q_norm_w", "arg2": "q_norm_bias", "arg3": "q_norm_t"},
    9: {"arg0": "q_f16", "arg2": "q_f32"},
    10: {"arg0": "kvCache", "arg2": "position_ids"},
    11: {"arg0": "position_ids", "arg1": "inv_freq", "arg2": "cos_s"},
    12: {"arg0": "cos_s", "arg1": "cos_b"},
    13: {"arg0": "position_ids", "arg1": "inv_freq", "arg2": "sin_s"},
    14: {"arg0": "sin_s", "arg1": "sin_b"},
    15: {"arg0": "q", "arg1": "cos_b", "arg2": "sin_b", "arg3": "q_rope"},
    16: {"arg0": "q_rope", "arg2": "q_attn"},
    17: {"arg0": "normed_hidden", "arg1": "kv_input"},
    18: {"arg0": "kv_input", "arg1": "v_w", "arg2": "v_partial"},
    19: {"arg0": "v_view", "arg2": "v_cache"},
    20: {"arg0": "residual0", "arg1": "k_norm_w", "arg2": "k_norm_bias", "arg3": "k_w", "arg4": "k_proj"},
    21: {"arg0": "k_view", "arg1": "k_norm_w", "arg2": "k_norm_bias", "arg3": "k_norm_t"},
    22: {"arg0": "k_f16", "arg2": "k_f32"},
    23: {"arg0": "k", "arg1": "cos_s", "arg2": "sin_s", "arg3": "k_rope"},
    24: {"arg0": "k_rope", "arg2": "k_cache"},
    25: {"arg0": "k_cache", "arg1": "kvCache"},
    26: {"arg0": "v_cache", "arg1": "kvCache"},
    27: {"arg0": "q_attn", "arg1": "kvCache", "arg2": "workspace", "arg3": "scale", "arg4": "attn_out"},
    28: {"arg0": "attn_out", "arg2": "o_proj_in"},
    29: {"arg0": "o_proj_in", "arg1": "o_w", "arg2": "o_partial"},
    30: {"arg0": "o_partial", "arg1": "o_s_b"},
    31: {"arg0": "post_norm_w", "arg1": "post_norm_bias", "arg2": "residual0", "arg3": "o_s_b", "arg5": "post_norm_s", "arg6": "residual1_s"},
    32: {"arg0": "residual1_s", "arg1": "residual1"},
    33: {"arg0": "post_norm_s", "arg1": "post_norm"},
    34: {"arg0": "post_norm", "arg1": "gate_w", "arg2": "up_w", "arg3": "down_w", "arg4": "mlp_partial"},
    35: {"arg0": "layer1_norm_w", "arg1": "layer1_norm_bias", "arg2": "residual1", "arg3": "mlp_partial", "arg5": "layer1_pre_norm", "arg6": "layer0_out"},
    36: {"arg0": "layer0_out", "arg1": "layer1_norm_w", "arg2": "layer1_norm_bias", "arg3": "layer1_q_w", "arg4": "layer1_q"},
}


MAIN1_OFF_IO_KEYS = {
    0: {"in": ("arg1",), "out": ("arg0",)},
    1: {"in": ("arg0",), "param": ("arg1",), "out": ("arg3",)},
    2: {"in": ("arg1",), "out": ("arg0",)},
    3: {"in": ("arg1",), "param": ("arg0",), "out": ("arg2",)},
    4: {"in": ("arg0",), "out": ("arg1",)},
    5: {"in": ("arg5", "arg3"), "param": ("arg0", "arg1", "arg2", "arg4"), "out": ("arg7", "arg8", "arg9")},
    6: {"in": ("arg0",), "out": ("arg1",)},
    7: {"in": ("arg0",), "param": ("arg1",), "out": ("arg3",)},
    8: {"in": ("arg0",), "param": ("arg1", "arg2"), "out": ("arg4",)},
    9: {"in": ("arg0",), "out": ("arg2",)},
    10: {"in": ("arg0",), "param": ("arg1",), "out": ("arg3", "arg4")},
    11: {"in": ("arg0",), "out": ("arg1",)},
    12: {"in": ("arg0",), "out": ("arg2",)},
    13: {"in": ("arg0",), "out": ("arg1",)},
    14: {"in": ("arg0", "arg1", "arg2"), "out": ("arg3",)},
    15: {"in": ("arg0",), "out": ("arg2",)},
    16: {"in": ("arg0",), "out": ("arg1",)},
    17: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    18: {"in": ("arg0",), "out": ("arg2",)},
    19: {"in": ("arg0",), "param": ("arg1", "arg2"), "out": ("arg4",)},
    20: {"in": ("arg0", "arg1", "arg2"), "out": ("arg3",)},
    21: {"in": ("arg0",), "out": ("arg2",)},
    22: {"in": ("arg0", "arg1"), "out": ("arg1",)},
    23: {"in": ("arg0", "arg1"), "out": ("arg1",)},
    24: {"in": ("arg0", "arg1"), "param": ("arg2", "arg3"), "out": ("arg4",)},
    25: {"in": ("arg0",), "out": ("arg2",)},
    26: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    27: {"in": ("arg0",), "out": ("arg1",)},
    28: {"in": ("arg2", "arg3"), "param": ("arg0", "arg1"), "out": ("arg5", "arg6")},
    29: {"in": ("arg0",), "out": ("arg1",)},
    30: {"in": ("arg0",), "out": ("arg1",)},
    31: {"in": ("arg0", "arg2"), "param": ("arg1", "arg3"), "out": ("arg5",)},
    32: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    33: {"in": ("arg3", "arg4"), "param": ("arg0", "arg1", "arg2"), "out": ("arg6", "arg7", "arg8")},
    34: {"in": ("arg0",), "param": ("arg1", "arg2"), "out": ("arg4",)},
}


MAIN1_COMPUTE_CCL_IO_KEYS = {
    0: {"in": ("arg1",), "out": ("arg0",)},
    1: {"in": ("arg0",), "param": ("arg1",), "out": ("arg3",)},
    2: {"in": ("arg1",), "out": ("arg0",)},
    3: {"in": ("arg1",), "param": ("arg0",), "out": ("arg2",)},
    4: {"in": ("arg0",), "out": ("arg1",)},
    5: {"in": ("arg4", "arg2"), "param": ("arg0", "arg1", "arg3"), "out": ("arg6", "arg7")},
    6: {"in": ("arg0",), "out": ("arg1",)},
    7: {"in": ("arg0",), "param": ("arg1",), "out": ("arg3",)},
    8: {"in": ("arg0",), "param": ("arg1", "arg2"), "out": ("arg3",)},
    9: {"in": ("arg0",), "out": ("arg2",)},
    10: {"in": ("arg0",), "out": ("arg2",)},
    11: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    12: {"in": ("arg0",), "out": ("arg1",)},
    13: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    14: {"in": ("arg0",), "out": ("arg1",)},
    15: {"in": ("arg0", "arg1", "arg2"), "out": ("arg3",)},
    16: {"in": ("arg0",), "out": ("arg2",)},
    17: {"in": ("arg0",), "out": ("arg1",)},
    18: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    19: {"in": ("arg0",), "out": ("arg2",)},
    20: {"in": ("arg0",), "param": ("arg1", "arg2", "arg3"), "out": ("arg4",)},
    21: {"in": ("arg0",), "param": ("arg1", "arg2"), "out": ("arg3",)},
    22: {"in": ("arg0",), "out": ("arg2",)},
    23: {"in": ("arg0", "arg1", "arg2"), "out": ("arg3",)},
    24: {"in": ("arg0",), "out": ("arg2",)},
    25: {"in": ("arg0", "arg1"), "out": ("arg1",)},
    26: {"in": ("arg0", "arg1"), "out": ("arg1",)},
    27: {"in": ("arg0", "arg1"), "param": ("arg2", "arg3"), "out": ("arg4",)},
    28: {"in": ("arg0",), "out": ("arg2",)},
    29: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    30: {"in": ("arg0",), "out": ("arg1",)},
    31: {"in": ("arg2", "arg3"), "param": ("arg0", "arg1"), "out": ("arg5", "arg6")},
    32: {"in": ("arg0",), "out": ("arg1",)},
    33: {"in": ("arg0",), "out": ("arg1",)},
    34: {"in": ("arg0",), "param": ("arg1", "arg2", "arg3"), "out": ("arg4",)},
    35: {"in": ("arg2", "arg3"), "param": ("arg0", "arg1"), "out": ("arg5", "arg6")},
    36: {"in": ("arg0",), "param": ("arg1", "arg2", "arg3"), "out": ("arg4",)},
}


MAIN1_OFF_HF_MEANING = {ordinal: "Qwen3DecoderLayer[0] main_segment_1_prim" for ordinal in range(35)}
MAIN1_OFF_HF_MEANING.update(
    {
        0: "Qwen3Model.forward input_ids sequence-sharded load",
        1: "Qwen3Model.forward input_ids mask",
        2: "Qwen3Model.forward input_ids replicated load",
        3: "Qwen3Model.embed_tokens",
        4: "Qwen3Model.embed_tokens all-to-all reshard",
        5: "Qwen3Model.embed_tokens materialization + Qwen3DecoderLayer.input_layernorm",
        6: "Qwen3DecoderLayer.input_layernorm materialization",
        7: "Qwen3Attention.q_proj",
        8: "Qwen3Attention.q_norm",
        9: "Qwen3RotaryEmbedding.forward position_ids",
        10: "Qwen3RotaryEmbedding.forward cos/angle",
        11: "Qwen3RotaryEmbedding.forward cos reshard",
        12: "Qwen3RotaryEmbedding.forward sin",
        13: "Qwen3RotaryEmbedding.forward sin reshard",
        14: "apply_rotary_pos_emb(q)",
        15: "Q layout for eager_attention_forward",
        16: "K/V normalized hidden materialization",
        17: "Qwen3Attention.v_proj",
        18: "V cache layout",
        19: "Qwen3Attention.k_norm",
        20: "apply_rotary_pos_emb(k)",
        21: "K cache layout",
        22: "Cache.update(k)",
        23: "Cache.update(v)",
        24: "eager_attention_forward / paged attention",
        25: "attention output layout",
        26: "Qwen3Attention.o_proj",
        27: "attention output reshard",
        28: "attention residual add + post_attention_layernorm",
        29: "attention residual materialization",
        30: "post_attention_layernorm materialization",
        31: "Qwen3MLP.gate_proj + Qwen3MLP.up_proj + Qwen3MLP.act_fn + mul",
        32: "Qwen3MLP.down_proj",
        33: "MLP residual add; layer0 output; layer1 q side output",
        34: "layer1 boundary: Qwen3Attention.q_norm",
    }
)


MAIN1_COMPUTE_CCL_HF_MEANING = {ordinal: "Qwen3DecoderLayer[0] main_segment_1_prim" for ordinal in range(37)}
MAIN1_COMPUTE_CCL_HF_MEANING.update(
    {
        0: "Qwen3Model.forward input_ids sequence-sharded load",
        1: "Qwen3Model.forward input_ids mask",
        2: "Qwen3Model.forward input_ids replicated load",
        3: "Qwen3Model.embed_tokens",
        4: "Qwen3Model.embed_tokens all-to-all reshard",
        5: "Qwen3Model.embed_tokens materialization + Qwen3DecoderLayer.input_layernorm",
        6: "Qwen3DecoderLayer.input_layernorm materialization",
        7: "Qwen3Attention.q_proj",
        8: "Qwen3Attention.q_norm",
        9: "Qwen3Attention.q_norm cast",
        10: "Qwen3RotaryEmbedding.forward position_ids",
        11: "Qwen3RotaryEmbedding.forward cos",
        12: "Qwen3RotaryEmbedding.forward cos reshard",
        13: "Qwen3RotaryEmbedding.forward sin",
        14: "Qwen3RotaryEmbedding.forward sin reshard",
        15: "apply_rotary_pos_emb(q)",
        16: "Q layout for eager_attention_forward",
        17: "K/V normalized hidden materialization",
        18: "Qwen3Attention.v_proj",
        19: "V cache layout",
        20: "Qwen3DecoderLayer.input_layernorm + Qwen3Attention.k_proj",
        21: "Qwen3Attention.k_norm",
        22: "Qwen3Attention.k_norm cast",
        23: "apply_rotary_pos_emb(k)",
        24: "K cache layout",
        25: "Cache.update(k)",
        26: "Cache.update(v)",
        27: "eager_attention_forward / paged attention",
        28: "attention output layout",
        29: "Qwen3Attention.o_proj",
        30: "attention output reshard",
        31: "attention residual add + post_attention_layernorm",
        32: "attention residual materialization",
        33: "post_attention_layernorm materialization",
        34: "Qwen3MLP.gate_proj + Qwen3MLP.up_proj + Qwen3MLP.act_fn + mul + Qwen3MLP.down_proj",
        35: "MLP residual add; layer0 output",
        36: "layer1 boundary: Qwen3DecoderLayer[1].input_layernorm",
    }
)


OFF_SPEC = ModeSpec(
    mode="off",
    title_mode="off",
    stem="qwen3-off-layer0",
    meta_path=p("tests_output/qwen3_reshard_off_probe2/cuda_admission/pe_16/CodeGen/cuda/cuda_meta.json"),
    launch_summary_path=p("tests_output/qwen3_reshard_off_probe2/cuda_admission/pe_16/CodeGen/cuda/launch_summary.txt"),
    triton_module_path=p("tests_output/qwen3_reshard_off_probe2/cuda_admission/pe_16/CodeGen/cuda/triton_module.py"),
    function_name="main_segment_1_prim",
    ordinals=tuple(range(35)),
    layer0_end=33,
    boundary=34,
    layout_rows=MAIN1_OFF_LAYOUT_ROWS,
    nn_data_edges=MAIN1_OFF_NN_DATA_EDGES,
    nn_skip_edges=frozenset(((5, 28), (29, 33))),
    hf_to_nn_mapping=MAIN1_OFF_HF_TO_NN_MAPPING,
    arg_roles=MAIN1_OFF_ARG_ROLES,
    io_keys=MAIN1_OFF_IO_KEYS,
    hf_meaning=MAIN1_OFF_HF_MEANING,
    helpers={},
)


COMPUTE_CCL_SPEC = ModeSpec(
    mode="compute-ccl",
    title_mode="compute-ccl",
    stem="qwen3-compute-ccl-layer0",
    meta_path=p("tests_output/test_qwen3_cuda_poc/cuda_admission/pe_16/CodeGen/cuda/cuda_meta.json"),
    launch_summary_path=p("tests_output/test_qwen3_cuda_poc/cuda_admission/pe_16/CodeGen/cuda/launch_summary.txt"),
    triton_module_path=p("tests_output/test_qwen3_cuda_poc/cuda_admission/pe_16/CodeGen/cuda/triton_module.py"),
    function_name="main_segment_1_prim",
    ordinals=OFF_SPEC.ordinals,
    layer0_end=OFF_SPEC.layer0_end,
    boundary=OFF_SPEC.boundary,
    layout_rows=OFF_SPEC.layout_rows,
    nn_data_edges=OFF_SPEC.nn_data_edges,
    nn_skip_edges=OFF_SPEC.nn_skip_edges,
    hf_to_nn_mapping=OFF_SPEC.hf_to_nn_mapping,
    arg_roles=OFF_SPEC.arg_roles,
    io_keys=OFF_SPEC.io_keys,
    hf_meaning=OFF_SPEC.hf_meaning,
    helpers={},
)


SPECS = {
    "off": OFF_SPEC,
    "compute-ccl": COMPUTE_CCL_SPEC,
}


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_inputs(spec: ModeSpec) -> tuple[dict, dict, list[dict], str, str]:
    meta = read_json(spec.meta_path)
    config = read_json(CONFIG_PATH)
    function = next(f for f in meta["functions"] if f["name"] == spec.function_name)
    launch_lookup = {launch["ordinal"]: launch for launch in function["launches"]}
    launches = [launch_lookup[ordinal] for ordinal in spec.ordinals]
    launch_summary = spec.launch_summary_path.read_text(encoding="utf-8-sig")
    triton_module = spec.triton_module_path.read_text(encoding="utf-8-sig")
    validate_inputs(spec, meta, launches, launch_summary, triton_module)
    return meta, config, launches, launch_summary, triton_module


def validate_inputs(
    spec: ModeSpec, meta: dict, launches: list[dict], launch_summary: str, triton_module: str
) -> None:
    if meta.get("pe_count") != 16:
        raise ValueError(f"{spec.mode}: expected pe_count=16, got {meta.get('pe_count')!r}")
    if meta.get("fused_kernel") != spec.mode:
        raise ValueError(f"{spec.mode}: expected fused_kernel={spec.mode}, got {meta.get('fused_kernel')!r}")
    ordinals = [launch["ordinal"] for launch in launches]
    if ordinals != list(spec.ordinals):
        raise ValueError(f"{spec.mode}: expected ordinals {list(spec.ordinals)}, got {ordinals}")
    if f"module pe_count=16 fused_kernel={spec.mode}" not in launch_summary:
        raise ValueError(f"{spec.mode}: launch_summary.txt does not describe PE=16 {spec.mode} mode")
    names = [launch["op_name"] for launch in launches]
    missing = [prefix for prefix in spec.required_fusions if not any(name.startswith(prefix) for name in names)]
    if missing:
        raise ValueError(f"{spec.mode}: missing required fusion(s): {', '.join(missing)}")
    if spec.mode == "compute-ccl" and any(name.startswith("fusion.cuda.") for name in names):
        raise ValueError("compute-ccl is CCL-only and should not contain fusion.cuda.* compute launches")
    for helper in sorted(set(spec.helpers.values())):
        for item in helper.split(" + "):
            if item.startswith("_nncase_") and item not in triton_module:
                raise ValueError(f"{spec.mode}: missing Triton helper {item}")


def node_id(ordinal: int) -> str:
    return f"n_{ordinal:02d}"


def is_fusion(launch: dict) -> bool:
    return launch["op_name"].startswith("fusion.cuda.")


def op_attrs(launch: dict) -> dict:
    attrs = launch.get("op_attrs")
    return attrs if isinstance(attrs, dict) else {}


def is_elided_ccl_tail_launch(launch: dict) -> bool:
    return bool(op_attrs(launch).get("elided_by_ccl_tail"))


def ccl_tail_names(launch: dict) -> list[str]:
    tails = op_attrs(launch).get("ccl_tails", [])
    if not isinstance(tails, list):
        return []
    return [str(tail.get("op_name", "ccl_tail.grs")) for tail in tails if isinstance(tail, dict)]


def has_fusions(launches: list[dict]) -> bool:
    return any(is_fusion(launch) for launch in launches)


def compact_dist(arg: dict) -> str:
    text = base.summarize_dist(arg.get("distributed_type"), arg)
    text = text.replace("SBP=", "")
    text = text.replace("S(0)", "S0")
    text = text.replace("[sequence_length", "[S")
    return text


def sbp_of(arg: dict) -> str:
    text = base.summarize_dist(arg.get("distributed_type"), arg)
    match = re.search(r"SBP=(\(.+\))(?: Partial=True)?$", text)
    if not match:
        match = re.search(r", (\(.+\)), \[p:", arg.get("distributed_type", ""))
    return match.group(1).replace("S(0)", "S0") if match else ""


def is_partial(arg: dict) -> bool:
    return "Partial: True" in arg.get("distributed_type", "") or "Partial=True" in base.summarize_dist(
        arg.get("distributed_type"), arg
    )


def ccl_display_name(launch: dict) -> str:
    if launch["op_name"] == "tensor_load":
        return "input load"
    if launch["op_name"] == "paged_attention":
        return "paged attention"
    if launch["op_name"] != "gather_reduce_scatter":
        return launch["op_name"]

    args = launch["buffer_arguments"]
    input_arg = args.get("arg0") or args.get("input")
    output_arg = args.get("arg1") or args.get("output")
    if not input_arg or not output_arg:
        return "reshard"

    in_sbp = sbp_of(input_arg)
    out_sbp = sbp_of(output_arg)
    in_has_s = "S0" in in_sbp
    out_has_s = "S0" in out_sbp
    if is_partial(input_arg):
        return "reduce-scatter" if out_has_s else "all-reduce"
    if in_has_s and out_has_s and in_sbp != out_sbp:
        return "all-to-all reshard"
    if in_has_s and not out_has_s:
        return "all-gather reshard"
    if not in_has_s and out_has_s:
        return "scatter reshard"
    return "reshard"


def fusion_display_name(op_name: str) -> str:
    family = fusion_family(op_name).replace("fusion.cuda.", "")
    return f"fusion {family}"


def short_meaning(spec: ModeSpec, ordinal: int) -> str:
    text = spec.hf_meaning.get(ordinal, "")
    replacements = {
        "Qwen3Model.": "",
        "Qwen3DecoderLayer.": "",
        "Qwen3Attention.": "",
        "Qwen3RotaryEmbedding.": "",
        "Qwen3MLP.": "",
        "forward ": "",
        "eager_attention_forward / ": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text[:46]


def nn_node_style(spec: ModeSpec, launch: dict) -> dict[str, str]:
    if is_elided_ccl_tail_launch(launch):
        return {
            "shape": "diamond",
            "style": "filled,dashed",
            "fillcolor": "#f2f2f2",
            "width": "5.0",
            "height": "1.35",
        }
    if launch.get("requires_collective"):
        return {
            "shape": "diamond",
            "style": "filled",
            "fillcolor": "#ffeaea",
            "width": "5.0",
            "height": "1.35",
        }
    if is_fusion(launch):
        return {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#e8f5e9",
            "width": "3.75",
        }
    if spec.boundary is not None and launch["ordinal"] == spec.boundary:
        return {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#f2f2f2",
            "width": "3.75",
        }
    return {
        "shape": "box",
        "style": "rounded,filled",
        "fillcolor": "#fff4e5",
        "width": "3.75",
    }


def format_tensor_line(spec: ModeSpec, launch: dict, key: str, role: str, prefix: str) -> str:
    arg = launch["buffer_arguments"][key]
    dist = base.summarize_dist(arg.get("distributed_type"), arg)
    name = arg["name"]
    name_and_role = name if name == role else f"{name} {role}"
    return f"{prefix}: {name_and_role}: {dist}"


def wrap_op_name(op_name: str) -> list[str]:
    prefixes = (
        "fusion.cuda.layer_norm_matmul_",
        "fusion.cuda.layer_norm_transpose_",
        "fusion.cuda.matmul_silu_matmul_mul_matmul_",
    )
    for prefix in prefixes:
        if op_name.startswith(prefix):
            return [prefix[:-1], op_name[len(prefix) :]]
    if len(op_name) <= 36:
        return [op_name]
    return re.findall(r".{1,36}(?:_|$)", op_name) or [op_name]


def nn_label(spec: ModeSpec, launch: dict) -> list[str]:
    ordinal = launch["ordinal"]
    if is_elided_ccl_tail_launch(launch):
        title = f"ord{ordinal:02d} ccl tail: {ccl_display_name(launch)}"
    elif launch.get("requires_collective"):
        title = f"ord{ordinal:02d} ccl: {ccl_display_name(launch)}"
    elif is_fusion(launch):
        title = f"ord{ordinal:02d}: {fusion_display_name(launch['op_name'])}"
    elif launch["op_name"] == "matmul":
        title = f"ord{ordinal:02d}: matmul"
    else:
        title = f"ord{ordinal:02d}: {short_meaning(spec, ordinal)}"
    lines = [title]

    in_keys = spec.io_keys[ordinal].get("in", ())
    out_keys = spec.io_keys[ordinal].get("out", ())
    param_keys = spec.io_keys[ordinal].get("param", ())
    if in_keys:
        key = in_keys[0]
        role = spec.arg_roles[ordinal][key]
        lines.append(f"in {role}: {compact_dist(launch['buffer_arguments'][key])}")
    if param_keys and (launch["op_name"] in {"matmul", "gather"} or is_fusion(launch)):
        key = param_keys[0]
        role = spec.arg_roles[ordinal][key]
        lines.append(f"param {role}: {compact_dist(launch['buffer_arguments'][key])}")
    if out_keys:
        key = out_keys[-1]
        role = spec.arg_roles[ordinal][key]
        lines.append(f"out {role}: {compact_dist(launch['buffer_arguments'][key])}")
    return lines


def nn_table_io(spec: ModeSpec, launch: dict) -> str:
    ordinal = launch["ordinal"]
    parts = []
    for prefix in ("in", "param", "out"):
        for key in spec.io_keys[ordinal].get(prefix, ()):
            parts.append(html.escape(format_tensor_line(spec, launch, key, spec.arg_roles[ordinal][key], prefix)))
    return "<br>".join(parts)


def table_op_name(launch: dict) -> str:
    op_name = html.escape(launch["op_name"])
    if launch.get("requires_collective"):
        semantic = html.escape(ccl_display_name(launch))
        if semantic != op_name:
            op_name = f"`{semantic}`<br>`{op_name}`"
        else:
            op_name = f"`{op_name}`"
    else:
        op_name = f"`{op_name}`"
    tails = ccl_tail_names(launch)
    if tails:
        op_name += "<br>`" + "`, `".join(html.escape(name) for name in tails) + "`"
    if is_elided_ccl_tail_launch(launch):
        producer = html.escape(str(op_attrs(launch).get("ccl_tail_producer_ordinal", "?")))
        op_name += f"<br><em>elided by producer ord{producer}</em>"
    return op_name


def helper_for(spec: ModeSpec, launch: dict) -> str:
    ordinal = launch["ordinal"]
    if ordinal in spec.helpers:
        return spec.helpers[ordinal]
    op_name = launch["op_name"]
    if is_elided_ccl_tail_launch(launch):
        producer = op_attrs(launch).get("ccl_tail_producer_ordinal", "?")
        return f"elided; ccl_tail after ord{producer}"
    tails = ccl_tail_names(launch)
    if op_name == "paged_attention":
        return "_nncase_paged_flash_attention_rank3_kernel"
    if launch.get("requires_collective") or op_name == "tensor_load":
        helper = "_nncase_ccl_rank4_kernel"
        return helper if not tails else f"{helper} + {', '.join(tails)}"
    if op_name == "gather":
        return "_nncase_pe_gather_axis0_rank2_kernel"
    if op_name == "matmul":
        return "_nncase_pe_matmul_kernel"
    if op_name == "ro_pe" or op_name.startswith("fusion.cuda.rope"):
        return "_nncase_pe_rope_rank4_kernel"
    if op_name == "update_paged_attention_kvcache":
        return "_nncase_pe_update_kv_rank4_kernel"
    if op_name.startswith("fusion.cuda.layer_norm_matmul"):
        return "_nncase_pe_layer_norm_matmul_kernel"
    if op_name.startswith("fusion.cuda.layer_norm_transpose"):
        return "_nncase_pe_layer_norm_transpose_rank4_kernel"
    if op_name.startswith("fusion.cuda.mul_cos") or op_name.startswith("fusion.cuda.mul_sin"):
        return "_nncase_pe_mul_unary_rank4_kernel"
    if op_name.startswith("fusion.cuda.matmul_silu_matmul_mul_matmul"):
        return "_nncase_pe_matmul_silu_matmul_mul_matmul_kernel"
    helper = f"triton_module.py::{op_name}"
    return helper if not tails else f"{helper} + {', '.join(tails)}"


def hf_mappings_by_ordinal(spec: ModeSpec) -> dict[int, list[str]]:
    hf_lookup = {node.node_id: node.lines[0] for node in base.HF_NODES}
    by_ordinal: dict[int, list[str]] = {}
    for hf_id, ordinal in spec.hf_to_nn_mapping:
        by_ordinal.setdefault(ordinal, []).append(hf_lookup[hf_id])
    return by_ordinal


def render_dot(spec: ModeSpec, launches: list[dict]) -> str:
    launch_lookup = {launch["ordinal"]: launch for launch in launches}
    hf_lookup = {node.node_id: node for node in base.HF_NODES}
    lines: list[str] = [
        f"digraph Qwen3{spec.title_mode.replace('-', '_')}Layer0 {{",
        '  graph [rankdir=TB, compound=true, splines=polyline, nodesep=0.34, ranksep=0.50,',
        '         fontsize=15, fontname="DejaVu Sans", labelloc=t,',
        f'         label="Qwen3-0.6B --fused-kernel={spec.title_mode} layer0, PE=16\\nsolid edges: selected data dependencies only; dashed cross edges: HF<->nncase semantic mapping"];',
        '  node [shape=box, style="rounded,filled", fontname="DejaVu Sans", fontsize=8.5,',
        '        color="#4f5661", margin="0.07,0.045"];',
        '  edge [fontname="DejaVu Sans", fontsize=8, color="#6a7380", arrowsize=0.60];',
        "",
        "  subgraph cluster_hf {",
        '    label="Hugging Face Qwen3 op names";',
        '    color="#8fb6df";',
        '    style="rounded";',
    ]
    for node in base.HF_NODES:
        lines.append(
            f'    {node.node_id} [label="{base.dot_label(node.lines)}", fillcolor="#eaf4ff", width=2.95, group="hf"];'
        )
    for idx, row in enumerate(spec.layout_rows):
        if not row.hf:
            lines.append(f'    hb_{idx:02d} [shape=point, style=invis, width=0.02, height=0.02, label=""];')
    lines.extend(
        [
            "  }",
            "",
            "  subgraph cluster_nncase {",
            '    label="nncase lowered ops, one node per ordinal";',
            '    color="#dfa85e";',
            '    style="rounded";',
        ]
    )
    for ordinal in spec.ordinals:
        launch = launch_lookup[ordinal]
        style = nn_node_style(spec, launch)
        attrs = {
            "label": base.dot_label(nn_label(spec, launch), left=not launch.get("requires_collective")),
            "group": "nn",
            **style,
        }
        attr_text = ", ".join(
            f'{key}="{value if key == "label" else base.dot_escape(value)}"' for key, value in attrs.items()
        )
        lines.append(f"    {node_id(ordinal)} [{attr_text}];")
    for idx, row in enumerate(spec.layout_rows):
        if not row.nn:
            lines.append(f'    nb_{idx:02d} [shape=point, style=invis, width=0.02, height=0.02, label=""];')
    lines.extend(["  }", ""])

    for idx, _row in enumerate(spec.layout_rows):
        lines.append(
            f'  sep_{idx:02d} [shape=point, label="", width=0.025, height=0.025, color="#777777", group="sep"];'
        )

    lines.append("")
    for idx, row in enumerate(spec.layout_rows):
        hf_nodes = row.hf or (f"hb_{idx:02d}",)
        nn_nodes = row.nn or (f"nb_{idx:02d}",)
        rank_nodes = [*hf_nodes, f"sep_{idx:02d}", *nn_nodes]
        lines.append("  { rank=same; " + "; ".join(rank_nodes) + "; }")
        ordered = [*hf_nodes, f"sep_{idx:02d}", *nn_nodes]
        for left, right in zip(ordered, ordered[1:]):
            lines.append(f"  {left} -> {right} [style=invis, weight=70];")

    lines.append("")
    for idx in range(len(spec.layout_rows) - 1):
        lines.append(
            f'  sep_{idx:02d} -> sep_{idx + 1:02d} '
            '[style=dashed, color="#777777", arrowhead=none, penwidth=1.15, weight=35];'
        )

    lines.append("\n  // Hugging Face data dependencies.")
    for src, dst in base.HF_DATA_EDGES:
        lines.append(f'  {src} -> {dst} [color="#527da8", penwidth=1.15];')

    lines.append("\n  // nncase data dependencies. Auxiliary const/mask/dim_var edges are intentionally omitted.")
    for src, dst in spec.nn_data_edges:
        attrs = 'color="#a66f24", penwidth=1.15'
        if (src, dst) in spec.nn_skip_edges:
            attrs += ', constraint=false, weight=0.2'
        lines.append(f"  {node_id(src)} -> {node_id(dst)} [{attrs}];")

    lines.append("\n  // Dashed semantic mapping between Hugging Face ops and lowered nncase ops.")
    for hf_id, ordinal in spec.hf_to_nn_mapping:
        lines.append(
            f'  {hf_id} -> {node_id(ordinal)} '
            '[style=dashed, color="#8a8f98", arrowhead=none, constraint=false, penwidth=0.85];'
        )

    lines.append("}")
    return "\n".join(lines) + "\n"


def segment1_ord04_summary(meta: dict) -> str:
    try:
        func = next(f for f in meta["functions"] if f["name"] == "main_segment_1_prim")
        launch = next(l for l in func["launches"] if l["ordinal"] == 4)
        args = launch["buffer_arguments"]
        in_type = base.summarize_dist(args["arg0"].get("distributed_type"), args["arg0"])
        out_type = base.summarize_dist(args["arg1"].get("distributed_type"), args["arg1"])
        return f"`main_segment_1_prim` ord04: `{launch['op_name']}` `{in_type}` -> `{out_type}`"
    except Exception as exc:  # pragma: no cover - diagnostic note only
        return f"`main_segment_1_prim` ord04: unavailable ({exc})"


def render_markdown(spec: ModeSpec, meta: dict, config: dict, launches: list[dict]) -> str:
    by_ordinal = hf_mappings_by_ordinal(spec)
    rows = []
    for launch in launches:
        ordinal = launch["ordinal"]
        hf_mapping = "<br>".join(
            html.escape(name) for name in by_ordinal.get(ordinal, [spec.hf_meaning[ordinal]])
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    str(ordinal),
                    table_op_name(launch),
                    hf_mapping,
                    nn_table_io(spec, launch),
                    f"`{helper_for(spec, launch)}`",
                ]
            )
            + " |"
        )

    collective_ordinals = [str(launch["ordinal"]) for launch in launches if launch.get("requires_collective")]
    fusion_names = sorted({fusion_family(launch["op_name"]) for launch in launches if is_fusion(launch)})
    fusion_section = (
        [f"No `fusion.cuda.*` ops are present in this `{spec.title_mode}` graph."]
        if not fusion_names
        else [f"- `{name}`" for name in fusion_names]
    )
    ccl_tail_section = []
    for launch in launches:
        tails = ccl_tail_names(launch)
        if tails:
            ccl_tail_section.append(f"- ord `{launch['ordinal']}` producer carries `{', '.join(tails)}`")
        if is_elided_ccl_tail_launch(launch):
            producer = op_attrs(launch).get("ccl_tail_producer_ordinal", "?")
            ccl_tail_section.append(f"- ord `{launch['ordinal']}` `{launch['op_name']}` is elided by producer ord `{producer}`")
    if not ccl_tail_section:
        ccl_tail_section = ["No opportunistic CCL tail sites are present in this layer0 slice."]
    boundary_line = (
        f"- ordinal `{spec.boundary}` is shown only as the layer1 boundary"
        if spec.boundary is not None
        else "- no extra layer1 boundary ordinal is shown"
    )

    md_lines = [
        f"# Qwen3 `--fused-kernel={spec.title_mode}` Layer0 Graph",
        "",
        f"This document maps the PE=16 `{spec.title_mode}` Qwen3 demo from `input_ids`,",
        "through embedding, to the end of decoder layer 0. It uses the same visual",
        "rules as the existing compute-mode graph: nncase ops are not merged, solid",
        "edges are only selected data dependencies, and dashed cross-column edges are",
        "semantic HF-to-nncase mappings.",
        "",
        "Source artifacts:",
        "",
        f"- `{spec.meta_path.relative_to(ROOT)}`",
        f"- `{spec.launch_summary_path.relative_to(ROOT)}`",
        f"- `{spec.triton_module_path.relative_to(ROOT)}`",
        f"- `{CONFIG_PATH.relative_to(ROOT)}`",
        f"- Hugging Face Qwen3 op names: [{HF_QWEN3_SOURCE_URL}]({HF_QWEN3_SOURCE_URL})",
        "",
        "Metadata checks:",
        "",
        "- `pe_count=16`",
        f"- `fused_kernel={spec.mode}`",
        f"- layer0 scope: `{spec.function_name}` ordinal `{spec.ordinals[0]}..{spec.layer0_end}`",
        boundary_line,
        "",
        "## Model Dimensions",
        "",
        "| Symbol | Meaning | Value |",
        "| --- | --- | --- |",
        "| `S` | dynamic `sequence_length` | runtime dynamic |",
        f"| `V` | vocabulary size | `{config['vocab_size']}` |",
        f"| `H` | hidden size | `{config['hidden_size']}` |",
        f"| `I` | MLP intermediate size | `{config['intermediate_size']}` |",
        f"| `QH` | query heads | `{config['num_attention_heads']}` |",
        f"| `KVH` | key/value heads | `{config['num_key_value_heads']}` |",
        f"| `D` | attention head dim | `{config['head_dim']}` |",
        "| `PE` | processing elements | `16` |",
        "",
        "## Graph",
        "",
        "Render command:",
        "",
        "```bash",
        "python docs/triton-backend/qwen3-layer0-fused-modes/generate_qwen3_layer0_fused_modes.py",
        "```",
        "",
        f'<img src="{spec.stem}.svg" alt="Qwen3 {spec.title_mode} layer0 graph" style="width: 100%; height: auto;">',
        "",
        "Legend:",
        "",
        "- blue rounded boxes: Hugging Face Qwen3 ops, with input -> output shapes",
        "- orange rounded boxes: ordinary nncase lowered ops",
        (
            "- green rounded boxes: `fusion.cuda.*` compute fused ops"
            if has_fusions(launches)
            else "- green rounded boxes: `fusion.cuda.*` compute fused ops (none in this graph)"
        ),
        f"- pink diamonds: `requires_collective=true` nncase launches ({', '.join(collective_ordinals)})",
        "- dashed grey diamonds: explicit CCL launches elided by a producer-side `ccl_tail.*.grs`",
        "- grey rounded box: layer1 boundary ordinal",
        "",
        "## Segment Note",
        "",
        f"- The graph above intentionally follows `{spec.function_name}` because this segment contains the requested ord04 all-to-all reshard.",
        f"- ord04 in this graph: {segment1_ord04_summary(meta)}",
        "- collective nodes use semantic display names in the graph; the launch table keeps the raw nncase op name.",
        "",
        "## Launch Table",
        "",
        "| Ord | nncase op | HF mapping | Input -> output dims/SBP | Triton helper |",
        "| --- | --- | --- | --- | --- |",
        *rows,
        "",
        "## Data-Dependency Policy",
        "",
        "The graph follows the selected `main data flow + key branch dependencies` policy.",
        "It keeps Q/K/V, RoPE, KV-cache update, paged attention, residual, and MLP data",
        "dependencies. Auxiliary constant, mask, workspace, and `dim_var` dependencies are",
        "kept in the table but not drawn as solid graph edges.",
        "",
        "## Fusions Present",
        "",
        *fusion_section,
        "",
        "## CCL Tail Sites",
        "",
        *ccl_tail_section,
        "",
        "## Boundary Note",
        "",
        f"Ordinal `{spec.layer0_end}` contains the layer0 main output. Any following ordinal",
        "is included only to mark the layer1 boundary or side output.",
        "",
    ]
    return "\n".join(md_lines)


def write_or_check(path: Path, text: str, check: bool) -> bool:
    if check:
        existing = path.read_text(encoding="utf-8") if path.exists() else None
        if existing != text:
            print(f"{path.relative_to(ROOT)} is not up to date", file=sys.stderr)
            return False
        return True
    path.write_text(text, encoding="utf-8")
    return True


def render_svg(dot: str) -> str:
    try:
        completed = subprocess.run(
            ["dot", "-Tsvg"],
            input=dot,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Graphviz `dot` is required to render the SVG") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or "Graphviz `dot` failed") from exc
    return re.sub(
        r'<svg width="[^"]+" height="[^"]+"',
        '<svg width="100%" height="auto"',
        completed.stdout,
        count=1,
    )


def fusion_family(op_name: str) -> str:
    prefixes = (
        "fusion.cuda.layer_norm_matmul",
        "fusion.cuda.layer_norm_transpose",
        "fusion.cuda.matmul_silu_matmul_mul_matmul",
        "fusion.cuda.mul_cos",
        "fusion.cuda.mul_sin",
        "fusion.cuda.rope",
    )
    for prefix in prefixes:
        if op_name.startswith(prefix):
            return prefix
    return op_name


def render_index(selected: list[ModeSpec]) -> str:
    selected_modes = {spec.mode for spec in selected}
    lines = [
        "# Qwen3 Layer0 Fused-Kernel Mode Graphs",
        "",
        "This directory contains PE=16 layer0 HF-to-nncase graphs generated with the same",
        "layout and visual conventions as `docs/triton-backend/qwen3-compute-layer0.*`.",
        "The off and compute-ccl graphs use `main_segment_1_prim`, where ordinal 04",
        "is the all-to-all `SBP=(B,S(0)) -> SBP=(S(0),B)` reshard record.",
        "",
        "`compute-ccl` is CCL-only in this directory: it does not contain",
        "`fusion.cuda.*` compute launches, and opportunistic GRS fusion is shown as",
        "`ccl_tail.<producer>.grs` metadata plus an elided explicit GRS ordinal.",
        "",
    ]
    for spec in selected:
        lines.extend(
            [
                f"- [`--fused-kernel={spec.title_mode}`](./{spec.stem}.md)",
                f"  - graph: [`{spec.stem}.svg`](./{spec.stem}.svg)",
            ]
        )
    lines.extend(
        [
            f"- [`vLLM TP=16 vs nncase --fused-kernel=off`](./{VLLM_COMPARE_STEM}.md)",
            f"  - graph: [`{VLLM_COMPARE_STEM}.svg`](./{VLLM_COMPARE_STEM}.svg)",
        ]
    )
    if {"off", "compute-ccl"}.issubset(selected_modes):
        lines.extend(
            [
                f"- [`off` vs `compute-ccl` CCL-tail comparison](./{COMPARE_STEM}.md)",
                f"  - graph: [`{COMPARE_STEM}.svg`](./{COMPARE_STEM}.svg)",
            ]
        )
    lines.extend(
        [
            "",
            "Regenerate:",
            "",
            "```bash",
            "python docs/triton-backend/qwen3-layer0-fused-modes/generate_qwen3_layer0_fused_modes.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def render_compare_dot(off_launches: list[dict], compute_ccl_launches: list[dict]) -> str:
    off_by_ordinal = {launch["ordinal"]: launch for launch in off_launches}
    ccl_by_ordinal = {launch["ordinal"]: launch for launch in compute_ccl_launches}
    ordinals = [
        ordinal
        for ordinal, launch in off_by_ordinal.items()
        if launch.get("op_name") == "gather_reduce_scatter"
    ]
    lines = [
        "digraph Qwen3OffVsComputeCclLayer0 {",
        '  graph [rankdir=LR, fontsize=15, fontname="DejaVu Sans", labelloc=t,',
        '         label="Qwen3-0.6B layer0: off explicit GRS vs compute-ccl CCL tails"];',
        '  node [shape=box, style="rounded,filled", fontname="DejaVu Sans", fontsize=9, color="#4f5661", margin="0.08,0.05"];',
        '  edge [fontname="DejaVu Sans", fontsize=8, color="#6a7380", arrowsize=0.60];',
        '  off [label="--fused-kernel=off\\nexplicit GRS launches", fillcolor="#ffeaea"];',
        '  cc [label="--fused-kernel=compute-ccl\\noff + opportunistic ccl_tail", fillcolor="#eaf4ff"];',
    ]
    for ordinal in ordinals:
        off_launch = off_by_ordinal[ordinal]
        ccl_launch = ccl_by_ordinal.get(ordinal)
        status = "missing"
        fill = "#fff4e5"
        if ccl_launch is not None and is_elided_ccl_tail_launch(ccl_launch):
            producer = op_attrs(ccl_launch).get("ccl_tail_producer_ordinal", "?")
            status = f"elided by producer ord{producer}"
            fill = "#e8f5e9"
        elif ccl_launch is not None:
            status = "still explicit"
            fill = "#ffeaea"
        label = base.dot_escape(f"ord{ordinal:02d}: {ccl_display_name(off_launch)}\\n{status}")
        lines.append(f'  grs_{ordinal:02d} [label="{label}", fillcolor="{fill}"];')
        lines.append(f"  off -> grs_{ordinal:02d} [style=invis, weight=20];")
        lines.append(f"  grs_{ordinal:02d} -> cc [style=invis, weight=20];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_compare_markdown(off_spec: ModeSpec, ccl_spec: ModeSpec, off_launches: list[dict], ccl_launches: list[dict]) -> str:
    ccl_by_ordinal = {launch["ordinal"]: launch for launch in ccl_launches}
    rows = []
    for launch in off_launches:
        if launch.get("op_name") != "gather_reduce_scatter":
            continue
        ordinal = launch["ordinal"]
        ccl_launch = ccl_by_ordinal.get(ordinal)
        if ccl_launch is None:
            status = "missing in compute-ccl metadata"
            producer = ""
        elif is_elided_ccl_tail_launch(ccl_launch):
            attrs = op_attrs(ccl_launch)
            status = "elided into CCL tail"
            producer = f"ord `{attrs.get('ccl_tail_producer_ordinal', '?')}` `{attrs.get('ccl_tail_producer_op', '?')}`"
        else:
            status = "still explicit"
            producer = ""
        rows.append(f"| {ordinal} | `{ccl_display_name(launch)}` | {status} | {producer} |")
    return "\n".join(
        [
            "# Qwen3 `off` vs `compute-ccl` Layer0 CCL Tail Comparison",
            "",
            "This comparison is generated from the same `tests_output` metadata as the",
            "per-mode graphs. `compute-ccl` is expected to match `off` for compute",
            "launches and differ only by opportunistic `gather_reduce_scatter` tail",
            "metadata plus elided explicit GRS ordinals.",
            "",
            "Source artifacts:",
            "",
            f"- `{off_spec.meta_path.relative_to(ROOT)}`",
            f"- `{ccl_spec.meta_path.relative_to(ROOT)}`",
            "",
            f'<img src="{COMPARE_STEM}.svg" alt="Qwen3 off vs compute-ccl CCL-tail comparison" style="width: 100%; height: auto;">',
            "",
            "| Ord | off GRS semantic | compute-ccl status | Producer |",
            "| --- | --- | --- | --- |",
            *rows,
            "",
        ]
    )


def generate_comparison(check: bool) -> bool:
    _off_meta, _off_config, off_launches, _off_summary, _off_triton = load_inputs(OFF_SPEC)
    _ccl_meta, _ccl_config, ccl_launches, _ccl_summary, _ccl_triton = load_inputs(COMPUTE_CCL_SPEC)
    dot = render_compare_dot(off_launches, ccl_launches)
    md = render_compare_markdown(OFF_SPEC, COMPUTE_CCL_SPEC, off_launches, ccl_launches)
    svg = render_svg(dot)
    ok = True
    ok &= write_or_check(OUT_DIR / f"{COMPARE_STEM}.dot", dot, check)
    ok &= write_or_check(OUT_DIR / f"{COMPARE_STEM}.md", md, check)
    ok &= write_or_check(OUT_DIR / f"{COMPARE_STEM}.svg", svg, check)
    return ok


def generate(spec: ModeSpec, check: bool) -> bool:
    meta, config, launches, _launch_summary, _triton_module = load_inputs(spec)
    dot = render_dot(spec, launches)
    md = render_markdown(spec, meta, config, launches)
    svg = render_svg(dot)
    ok = True
    ok &= write_or_check(spec.out_dot, dot, check)
    ok &= write_or_check(spec.out_md, md, check)
    ok &= write_or_check(spec.out_svg, svg, check)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify generated docs are up to date")
    parser.add_argument(
        "--mode",
        action="append",
        choices=sorted(SPECS),
        help="mode to generate; defaults to off and compute-ccl",
    )
    args = parser.parse_args()

    selected = [SPECS[name] for name in (args.mode or ["off", "compute-ccl"])]
    ok = True
    for spec in selected:
        ok &= generate(spec, args.check)
    if {"off", "compute-ccl"}.issubset({spec.mode for spec in selected}):
        ok &= generate_comparison(args.check)
    ok &= write_or_check(OUT_DIR / "README.md", render_index(selected), args.check)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
