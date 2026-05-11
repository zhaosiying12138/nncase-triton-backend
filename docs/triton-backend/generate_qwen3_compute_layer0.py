#!/usr/bin/env python3
"""Generate the Qwen3 PE=16 compute-mode layer0 HF/nncase graph."""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CUDA_DIR = ROOT / "tests_output/test_qwen3_cuda_poc/cuda_admission/pe_16/CodeGen/cuda"
META_PATH = CUDA_DIR / "cuda_meta.json"
LAUNCH_SUMMARY_PATH = CUDA_DIR / "launch_summary.txt"
TRITON_MODULE_PATH = CUDA_DIR / "triton_module.py"
CONFIG_PATH = ROOT / "tests/llm/Qwen/Qwen3-0.6B/config.json"

OUT_DOT = ROOT / "docs/triton-backend/qwen3-compute-layer0.dot"
OUT_SVG = ROOT / "docs/triton-backend/qwen3-compute-layer0.svg"
OUT_MD = ROOT / "docs/triton-backend/qwen3-compute-layer0.md"

HF_QWEN3_SOURCE_URL = (
    "https://github.com/huggingface/transformers/blob/main/"
    "src/transformers/models/qwen3/modeling_qwen3.py"
)


@dataclass(frozen=True)
class HfNode:
    node_id: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Row:
    hf: tuple[str, ...] = ()
    nn: tuple[str, ...] = ()


HF_NODES: tuple[HfNode, ...] = (
    HfNode(
        "hf_input",
        (
            "Qwen3Model.forward",
            "in: input_ids i64[S]",
            "out: hidden input ids i64[S]",
        ),
    ),
    HfNode(
        "hf_embed",
        (
            "Qwen3Model.embed_tokens",
            "in: input_ids i64[S]",
            "param: weight f16[V,1024]",
            "out: hidden_states f16[S,1024]",
        ),
    ),
    HfNode(
        "hf_input_ln",
        (
            "Qwen3DecoderLayer.input_layernorm",
            "in: hidden_states f16[S,1024]",
            "out: normed hidden f16[S,1024]",
            "eps=1e-06",
        ),
    ),
    HfNode(
        "hf_q_proj",
        (
            "Qwen3Attention.q_proj",
            "in: normed hidden f16[S,1024]",
            "param: weight f16[1024,2048]",
            "out: query_states f16[S,16,128]",
        ),
    ),
    HfNode(
        "hf_q_norm",
        (
            "Qwen3Attention.q_norm",
            "in: query_states f16[S,16,128]",
            "out: query_states f16[16,S,128]",
            "eps=1e-06",
        ),
    ),
    HfNode(
        "hf_rotary",
        (
            "Qwen3RotaryEmbedding.forward",
            "in: position_ids i64[S]",
            "param: inv_freq f32[128]",
            "out: cos/sin f32[S,128]",
        ),
    ),
    HfNode(
        "hf_rope_q",
        (
            "apply_rotary_pos_emb",
            "in: q f32[16,S,128] + cos/sin",
            "out: q_rope f32[16,S,128]",
        ),
    ),
    HfNode(
        "hf_v_proj",
        (
            "Qwen3Attention.v_proj",
            "in: normed hidden f16[S,1024]",
            "param: weight f16[1024,1024]",
            "out: value_states f16[S,8,128]",
        ),
    ),
    HfNode(
        "hf_k_proj",
        (
            "Qwen3Attention.k_proj",
            "in: normed hidden f16[S,1024]",
            "param: weight f16[1024,1024]",
            "out: key_states f16[S,8,128]",
        ),
    ),
    HfNode(
        "hf_k_norm",
        (
            "Qwen3Attention.k_norm",
            "in: key_states f16[S,8,128]",
            "out: key_states f32[8,S,128]",
            "eps=1e-06",
        ),
    ),
    HfNode(
        "hf_rope_k",
        (
            "apply_rotary_pos_emb",
            "in: k f32[8,S,128] + cos/sin",
            "out: k_rope f32[8,S,128]",
        ),
    ),
    HfNode(
        "hf_cache",
        (
            "Cache.update",
            "in: k/v f16[8,128,S]",
            "out: paged KV cache side effect",
        ),
    ),
    HfNode(
        "hf_attn",
        (
            "eager_attention_forward",
            "in: q f16[16,128,S] + KV cache",
            "out: attn_output f16[16,128,S]",
        ),
    ),
    HfNode(
        "hf_o_proj",
        (
            "Qwen3Attention.o_proj",
            "in: attn_output f16[S,2048]",
            "param: weight f16[2048,1024]",
            "out: attn_output f16[S,1024]",
        ),
    ),
    HfNode(
        "hf_attn_resid",
        (
            "Qwen3DecoderLayer.forward residual add",
            "in: hidden f16[S,1024] + attn f16[S,1024]",
            "out: residual1 f16[S,1024]",
        ),
    ),
    HfNode(
        "hf_post_ln",
        (
            "Qwen3DecoderLayer.post_attention_layernorm",
            "in: residual1 f16[S,1024]",
            "out: post_norm f16[S,1024]",
            "eps=1e-06",
        ),
    ),
    HfNode(
        "hf_gate",
        (
            "Qwen3MLP.gate_proj",
            "in: post_norm f16[S,1024]",
            "out: gate f16[S,3072]",
        ),
    ),
    HfNode(
        "hf_up",
        (
            "Qwen3MLP.up_proj",
            "in: post_norm f16[S,1024]",
            "out: up f16[S,3072]",
        ),
    ),
    HfNode(
        "hf_act_mul",
        (
            "Qwen3MLP.act_fn + mul",
            "in: gate/up f16[S,3072]",
            "out: hidden f16[S,3072]",
        ),
    ),
    HfNode(
        "hf_down",
        (
            "Qwen3MLP.down_proj",
            "in: hidden f16[S,3072]",
            "out: mlp f16[S,1024]",
        ),
    ),
    HfNode(
        "hf_mlp_resid",
        (
            "Qwen3DecoderLayer.forward residual add",
            "in: residual1 f16[S,1024] + mlp f16[S,1024]",
            "out: layer0_out f16[S,1024]",
        ),
    ),
    HfNode(
        "hf_layer1",
        (
            "Qwen3DecoderLayer[1].input_layernorm",
            "boundary only",
            "in: layer0_out f16[S,1024]",
        ),
    ),
)


LAYOUT_ROWS: tuple[Row, ...] = (
    Row(("hf_input",), ("n_00",)),
    Row((), ("n_01",)),
    Row(("hf_embed",), ("n_02",)),
    Row((), ("n_03",)),
    Row((), ("n_04",)),
    Row(("hf_input_ln",), ("n_05", "n_13")),
    Row(("hf_q_proj",), ()),
    Row(("hf_q_norm",), ("n_06", "n_14")),
    Row(("hf_v_proj",), ("n_07", "n_15")),
    Row(("hf_rotary",), ("n_08", "n_16")),
    Row((), ("n_09", "n_10")),
    Row(("hf_rope_q",), ("n_11", "n_17")),
    Row(("hf_k_proj",), ()),
    Row(("hf_k_norm",), ("n_12", "n_18")),
    Row((), ("n_19",)),
    Row(("hf_rope_k",), ("n_20",)),
    Row((), ("n_21",)),
    Row(("hf_cache",), ("n_22", "n_23")),
    Row(("hf_attn",), ("n_24",)),
    Row((), ("n_25",)),
    Row(("hf_o_proj",), ("n_26",)),
    Row(("hf_attn_resid",), ("n_27",)),
    Row(("hf_post_ln",), ()),
    Row(("hf_gate", "hf_up"), ("n_28",)),
    Row(("hf_act_mul",), ()),
    Row(("hf_down",), ()),
    Row(("hf_mlp_resid",), ("n_29",)),
    Row(("hf_layer1",), ("n_30",)),
)


HF_DATA_EDGES: tuple[tuple[str, str], ...] = (
    ("hf_input", "hf_embed"),
    ("hf_embed", "hf_input_ln"),
    ("hf_embed", "hf_rotary"),
    ("hf_input_ln", "hf_q_proj"),
    ("hf_input_ln", "hf_v_proj"),
    ("hf_input_ln", "hf_k_proj"),
    ("hf_q_proj", "hf_q_norm"),
    ("hf_q_norm", "hf_rope_q"),
    ("hf_rotary", "hf_rope_q"),
    ("hf_rotary", "hf_rope_k"),
    ("hf_v_proj", "hf_cache"),
    ("hf_k_proj", "hf_k_norm"),
    ("hf_k_norm", "hf_rope_k"),
    ("hf_rope_k", "hf_cache"),
    ("hf_rope_q", "hf_attn"),
    ("hf_cache", "hf_attn"),
    ("hf_attn", "hf_o_proj"),
    ("hf_o_proj", "hf_attn_resid"),
    ("hf_embed", "hf_attn_resid"),
    ("hf_attn_resid", "hf_post_ln"),
    ("hf_post_ln", "hf_gate"),
    ("hf_post_ln", "hf_up"),
    ("hf_gate", "hf_act_mul"),
    ("hf_up", "hf_act_mul"),
    ("hf_act_mul", "hf_down"),
    ("hf_down", "hf_mlp_resid"),
    ("hf_attn_resid", "hf_mlp_resid"),
    ("hf_mlp_resid", "hf_layer1"),
)


HF_SKIP_EDGES: frozenset[tuple[str, str]] = frozenset()


NN_DATA_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 11),
    (8, 9),
    (8, 10),
    (9, 11),
    (10, 11),
    (11, 12),
    (4, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 23),
    (14, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (9, 20),
    (10, 20),
    (20, 21),
    (21, 22),
    (12, 24),
    (22, 24),
    (23, 24),
    (24, 25),
    (25, 26),
    (26, 27),
    (4, 27),
    (27, 28),
    (27, 29),
    (28, 29),
    (29, 30),
)


NN_SKIP_EDGES: frozenset[tuple[int, int]] = frozenset(
    (
        (4, 27),
        (27, 29),
    )
)


HF_TO_NN_MAPPING: tuple[tuple[str, int], ...] = (
    ("hf_input", 0),
    ("hf_input", 1),
    ("hf_embed", 2),
    ("hf_embed", 3),
    ("hf_embed", 4),
    ("hf_input_ln", 5),
    ("hf_input_ln", 13),
    ("hf_q_proj", 5),
    ("hf_q_norm", 6),
    ("hf_q_norm", 7),
    ("hf_rotary", 8),
    ("hf_rotary", 9),
    ("hf_rotary", 10),
    ("hf_rope_q", 11),
    ("hf_rope_q", 12),
    ("hf_v_proj", 15),
    ("hf_v_proj", 16),
    ("hf_k_proj", 17),
    ("hf_k_norm", 18),
    ("hf_k_norm", 19),
    ("hf_rope_k", 20),
    ("hf_rope_k", 21),
    ("hf_cache", 22),
    ("hf_cache", 23),
    ("hf_attn", 24),
    ("hf_attn", 25),
    ("hf_o_proj", 26),
    ("hf_attn_resid", 27),
    ("hf_post_ln", 27),
    ("hf_gate", 28),
    ("hf_up", 28),
    ("hf_act_mul", 28),
    ("hf_down", 28),
    ("hf_mlp_resid", 29),
    ("hf_layer1", 30),
)


ARG_ROLES: dict[int, dict[str, str]] = {
    0: {"arg1": "input_ids", "arg0": "ids_local"},
    1: {"arg0": "input_ids", "arg1": "pad_id", "arg3": "mask"},
    2: {"arg0": "embed_w", "arg1": "input_ids", "arg2": "embed"},
    3: {"arg0": "mask", "arg1": "fill_value", "arg2": "embed", "arg4": "masked_embed"},
    4: {"arg0": "embed_shard", "arg1": "hidden"},
    5: {"arg0": "hidden", "arg1": "rms_w", "arg2": "rms_bias", "arg3": "q_w", "arg4": "q_proj"},
    6: {"arg0": "q_view", "arg1": "q_norm_w", "arg2": "q_norm_bias", "arg3": "q_norm_t"},
    7: {"arg0": "q_f16", "arg2": "q_f32"},
    8: {"arg0": "kvCache", "arg2": "position_ids"},
    9: {"arg0": "position_ids", "arg1": "inv_freq", "arg2": "cos"},
    10: {"arg0": "position_ids", "arg1": "inv_freq", "arg2": "sin"},
    11: {"arg0": "q", "arg1": "cos", "arg2": "sin", "arg3": "q_rope"},
    12: {"arg0": "q_rope", "arg2": "q_attn"},
    13: {"arg0": "hidden", "arg1": "rms_w", "arg2": "rms_bias", "arg4": "kv_norm"},
    14: {"arg0": "kv_norm", "arg1": "kv_input"},
    15: {"arg0": "kv_input", "arg1": "v_w", "arg2": "v_partial"},
    16: {"arg0": "v_view", "arg2": "v_cache"},
    17: {"arg0": "kv_input", "arg1": "k_w", "arg2": "k_partial"},
    18: {"arg0": "k_view", "arg1": "k_norm_w", "arg2": "k_norm_bias", "arg3": "k_norm_t"},
    19: {"arg0": "k_f16", "arg2": "k_f32"},
    20: {"arg0": "k", "arg1": "cos", "arg2": "sin", "arg3": "k_rope"},
    21: {"arg0": "k_rope", "arg2": "k_cache"},
    22: {"arg0": "k_cache", "arg1": "kvCache"},
    23: {"arg0": "v_cache", "arg1": "kvCache"},
    24: {"arg0": "q_attn", "arg1": "kvCache", "arg2": "workspace", "arg3": "scale", "arg4": "attn_out"},
    25: {"arg0": "attn_out", "arg2": "o_proj_in"},
    26: {"arg0": "o_proj_in", "arg1": "o_w", "arg2": "o_partial"},
    27: {
        "arg0": "post_norm_w",
        "arg1": "post_norm_bias",
        "arg2": "residual0",
        "arg3": "o_partial",
        "arg5": "post_norm",
        "arg6": "residual1",
    },
    28: {
        "arg0": "post_norm",
        "arg1": "gate_w",
        "arg2": "up_w",
        "arg3": "down_w",
        "arg4": "mlp_partial",
    },
    29: {
        "arg0": "layer1_norm_w",
        "arg1": "layer1_norm_bias",
        "arg2": "residual1",
        "arg3": "mlp_partial",
        "arg5": "layer1_pre_norm",
        "arg6": "layer0_out",
    },
    30: {
        "arg0": "layer0_out",
        "arg1": "layer1_norm_w",
        "arg2": "layer1_norm_bias",
        "arg3": "layer1_q_w",
        "arg4": "layer1_q",
    },
}


IO_KEYS: dict[int, dict[str, tuple[str, ...]]] = {
    0: {"in": ("arg1",), "out": ("arg0",)},
    1: {"in": ("arg0",), "param": ("arg1",), "out": ("arg3",)},
    2: {"in": ("arg1",), "param": ("arg0",), "out": ("arg2",)},
    3: {"in": ("arg0", "arg2"), "param": ("arg1",), "out": ("arg4",)},
    4: {"in": ("arg0",), "out": ("arg1",)},
    5: {"in": ("arg0",), "param": ("arg1", "arg2", "arg3"), "out": ("arg4",)},
    6: {"in": ("arg0",), "param": ("arg1", "arg2"), "out": ("arg3",)},
    7: {"in": ("arg0",), "out": ("arg2",)},
    8: {"in": ("arg0",), "out": ("arg2",)},
    9: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    10: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    11: {"in": ("arg0", "arg1", "arg2"), "out": ("arg3",)},
    12: {"in": ("arg0",), "out": ("arg2",)},
    13: {"in": ("arg0",), "param": ("arg1", "arg2"), "out": ("arg4",)},
    14: {"in": ("arg0",), "out": ("arg1",)},
    15: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    16: {"in": ("arg0",), "out": ("arg2",)},
    17: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    18: {"in": ("arg0",), "param": ("arg1", "arg2"), "out": ("arg3",)},
    19: {"in": ("arg0",), "out": ("arg2",)},
    20: {"in": ("arg0", "arg1", "arg2"), "out": ("arg3",)},
    21: {"in": ("arg0",), "out": ("arg2",)},
    22: {"in": ("arg0", "arg1"), "out": ("arg1",)},
    23: {"in": ("arg0", "arg1"), "out": ("arg1",)},
    24: {"in": ("arg0", "arg1"), "param": ("arg2", "arg3"), "out": ("arg4",)},
    25: {"in": ("arg0",), "out": ("arg2",)},
    26: {"in": ("arg0",), "param": ("arg1",), "out": ("arg2",)},
    27: {"in": ("arg2", "arg3"), "param": ("arg0", "arg1"), "out": ("arg5", "arg6")},
    28: {"in": ("arg0",), "param": ("arg1", "arg2", "arg3"), "out": ("arg4",)},
    29: {"in": ("arg2", "arg3"), "param": ("arg0", "arg1"), "out": ("arg5", "arg6")},
    30: {"in": ("arg0",), "param": ("arg1", "arg2", "arg3"), "out": ("arg4",)},
}


HF_MEANING: dict[int, str] = {
    0: "Qwen3Model.forward input_ids",
    1: "Qwen3Model.forward input_ids mask",
    2: "Qwen3Model.embed_tokens",
    3: "Qwen3Model.embed_tokens materialization",
    4: "Qwen3Model.embed_tokens materialization",
    5: "Qwen3DecoderLayer.input_layernorm + Qwen3Attention.q_proj",
    6: "Qwen3Attention.q_norm + view/transpose",
    7: "Qwen3Attention.q_norm dtype/layout cast",
    8: "Qwen3RotaryEmbedding.forward position_ids",
    9: "Qwen3RotaryEmbedding.forward cos",
    10: "Qwen3RotaryEmbedding.forward sin",
    11: "apply_rotary_pos_emb(q)",
    12: "Q layout for eager_attention_forward",
    13: "Qwen3DecoderLayer.input_layernorm for K/V",
    14: "K/V normalized hidden materialization",
    15: "Qwen3Attention.v_proj",
    16: "V cache layout",
    17: "Qwen3Attention.k_proj",
    18: "Qwen3Attention.k_norm + view/transpose",
    19: "Qwen3Attention.k_norm dtype/layout cast",
    20: "apply_rotary_pos_emb(k)",
    21: "K cache layout",
    22: "Cache.update(k)",
    23: "Cache.update(v)",
    24: "eager_attention_forward / paged attention",
    25: "attention output layout",
    26: "Qwen3Attention.o_proj",
    27: "attention residual add + Qwen3DecoderLayer.post_attention_layernorm",
    28: "Qwen3MLP.gate_proj/up_proj/act_fn/mul/down_proj",
    29: "MLP residual add; layer0 output",
    30: "layer1 boundary: Qwen3DecoderLayer[1].input_layernorm + q_proj",
}


HELPERS: dict[int, str] = {
    0: "_nncase_ccl_rank4_kernel",
    1: "_nncase_pe_binary_rank4_kernel",
    2: "_nncase_pe_gather_axis0_rank2_kernel",
    3: "_nncase_pe_where_rank4_kernel",
    4: "_nncase_ccl_rank4_kernel",
    5: "_nncase_pe_layer_norm_matmul_kernel",
    6: "_nncase_pe_layer_norm_transpose_rank4_kernel",
    7: "_nncase_pe_copy_rank4_kernel",
    8: "_nncase_pe_position_ids_kernel",
    9: "_nncase_pe_mul_unary_rank4_kernel",
    10: "_nncase_pe_mul_unary_rank4_kernel",
    11: "_nncase_pe_rope_rank4_kernel",
    12: "_nncase_pe_transpose_rank4_kernel",
    13: "_nncase_pe_layer_norm_kernel",
    14: "_nncase_ccl_rank4_kernel",
    15: "_nncase_pe_matmul_kernel",
    16: "_nncase_pe_transpose_rank4_kernel",
    17: "_nncase_pe_matmul_kernel",
    18: "_nncase_pe_layer_norm_transpose_rank4_kernel",
    19: "_nncase_pe_copy_rank4_kernel",
    20: "_nncase_pe_rope_rank4_kernel",
    21: "_nncase_pe_transpose_rank4_kernel",
    22: "_nncase_pe_update_kv_rank4_kernel",
    23: "_nncase_pe_update_kv_rank4_kernel",
    24: "_nncase_paged_flash_attention_rank3_kernel",
    25: "_nncase_pe_transpose_rank4_kernel",
    26: "_nncase_pe_matmul_kernel",
    27: "_nncase_pe_binary_rank4_kernel + _nncase_pe_layer_norm_kernel",
    28: "_nncase_pe_matmul_silu_matmul_mul_matmul_kernel",
    29: "_nncase_pe_binary_rank4_kernel + _nncase_pe_layer_norm_kernel",
    30: "_nncase_pe_layer_norm_matmul_kernel",
}


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_inputs() -> tuple[dict, dict, list[dict], str, str]:
    meta = read_json(META_PATH)
    config = read_json(CONFIG_PATH)
    function = next(f for f in meta["functions"] if f["name"] == "main_segment_0_prim")
    launches = function["launches"][:31]
    launch_summary = LAUNCH_SUMMARY_PATH.read_text(encoding="utf-8-sig")
    triton_module = TRITON_MODULE_PATH.read_text(encoding="utf-8-sig")
    validate_inputs(meta, launches, launch_summary, triton_module)
    return meta, config, launches, launch_summary, triton_module


def validate_inputs(meta: dict, launches: list[dict], launch_summary: str, triton_module: str) -> None:
    if meta.get("pe_count") != 16:
        raise ValueError(f"expected pe_count=16, got {meta.get('pe_count')!r}")
    if meta.get("fused_kernel") != "compute":
        raise ValueError(f"expected fused_kernel=compute, got {meta.get('fused_kernel')!r}")
    ordinals = [launch["ordinal"] for launch in launches]
    if ordinals != list(range(31)):
        raise ValueError(f"expected main_segment_0_prim ordinals 0..30, got {ordinals}")
    if "module pe_count=16 fused_kernel=compute" not in launch_summary:
        raise ValueError("launch_summary.txt does not describe PE=16 compute mode")
    names = [launch["op_name"] for launch in launches]
    required_prefixes = (
        "fusion.cuda.layer_norm_matmul",
        "fusion.cuda.layer_norm_transpose",
        "fusion.cuda.mul_cos",
        "fusion.cuda.mul_sin",
        "fusion.cuda.rope",
        "fusion.cuda.matmul_silu_matmul_mul_matmul",
    )
    missing = [prefix for prefix in required_prefixes if not any(name.startswith(prefix) for name in names[:30])]
    if missing:
        raise ValueError(f"missing required compute fusion(s): {', '.join(missing)}")
    collective_ordinals = [launch["ordinal"] for launch in launches if launch.get("requires_collective")]
    if collective_ordinals != [0, 4, 14, 24]:
        raise ValueError(f"unexpected collective ordinal set: {collective_ordinals}")
    required_helpers = (
        "_nncase_pe_layer_norm_matmul_kernel",
        "_nncase_pe_layer_norm_transpose_rank4_kernel",
        "_nncase_pe_rope_rank4_kernel",
        "_nncase_paged_flash_attention_rank3_kernel",
        "_nncase_pe_matmul_silu_matmul_mul_matmul_kernel",
    )
    absent = [helper for helper in required_helpers if helper not in triton_module]
    if absent:
        raise ValueError(f"missing required Triton helper(s): {', '.join(absent)}")


def dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def dot_label(lines: list[str] | tuple[str, ...], *, left: bool = True) -> str:
    separator = "\\l" if left else "\\n"
    suffix = "\\l" if left else ""
    return separator.join(dot_escape(line) for line in lines) + suffix


def node_id(ordinal: int) -> str:
    return f"n_{ordinal:02d}"


def is_fusion(launch: dict) -> bool:
    return launch["op_name"].startswith("fusion.cuda.")


def nn_node_style(launch: dict) -> dict[str, str]:
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
    if launch["ordinal"] == 30:
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


def summarize_dist(distributed_type: str | None, arg: dict | None = None) -> str:
    if distributed_type:
        parts = distributed_type.rsplit(", ", 3)
        if len(parts) == 4:
            dtype_shape, sbp, placement, partial = parts
            dtype_shape = compact_shape(dtype_shape)
            partial_value = partial.split(": ", 1)[1]
            partial_suffix = " Partial=True" if partial_value == "True" else ""
            return f"{dtype_shape} SBP={sbp}{partial_suffix}"
        return compact_shape(distributed_type)
    if arg is None:
        return "unknown"
    dtype = compact_dtype(str(arg.get("dtype", "unknown")))
    shape = compact_shape("[" + ",".join(format_dim(d) for d in arg.get("shape", [])) + "]")
    if "PagedAttentionKVCache" in dtype:
        return "PagedAttentionKVCache"
    return f"{dtype}{shape}"


def sbp_of(arg: dict) -> str:
    text = summarize_dist(arg.get("distributed_type"), arg)
    match = re.search(r"SBP=(\(.+\))(?: Partial=True)?$", text)
    if not match:
        match = re.search(r", (\(.+\)), \[p:", arg.get("distributed_type", ""))
    return match.group(1).replace("S(0)", "S0") if match else ""


def is_partial(arg: dict) -> bool:
    return "Partial: True" in arg.get("distributed_type", "") or "Partial=True" in summarize_dist(
        arg.get("distributed_type"), arg
    )


def collective_display_name(launch: dict) -> str:
    op_name = launch["op_name"]
    if op_name != "gather_reduce_scatter":
        return op_name

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


def compact_dtype(dtype: str) -> str:
    return (
        dtype.replace("DataTypes.", "")
        .replace("Float16", "f16")
        .replace("Float32", "f32")
        .replace("Int64", "i64")
        .replace("UInt8", "u8")
        .replace("Boolean", "bool")
        .replace("new ReferenceType(new PagedAttentionKVCacheType())", "PagedAttentionKVCache")
    )


def compact_shape(text: str) -> str:
    replacements = {
        "sequence_length": "S",
        "151936": "V",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def format_dim(dim: dict) -> str:
    if dim.get("kind") == "dynamic":
        return dim["symbol"]
    if dim.get("kind") == "fixed":
        return str(dim["value"])
    return "?"


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


def format_tensor_line(launch: dict, ordinal: int, key: str, role: str, prefix: str) -> str:
    arg = launch["buffer_arguments"][key]
    dist = summarize_dist(arg.get("distributed_type"), arg)
    name = arg["name"]
    name_and_role = name if name == role else f"{name} {role}"
    return f"{prefix}: {name_and_role}: {dist}"


def nn_label(launch: dict) -> list[str]:
    ordinal = launch["ordinal"]
    lines = [f"ord{ordinal:02d} {launch['kind']}"]
    op_name = collective_display_name(launch) if launch.get("requires_collective") else launch["op_name"]
    lines.extend(wrap_op_name(op_name))
    for prefix in ("in", "param", "out"):
        for key in IO_KEYS[ordinal].get(prefix, ()):
            lines.append(format_tensor_line(launch, ordinal, key, ARG_ROLES[ordinal][key], prefix))
    return lines


def nn_table_io(launch: dict) -> str:
    ordinal = launch["ordinal"]
    parts = []
    for prefix in ("in", "param", "out"):
        for key in IO_KEYS[ordinal].get(prefix, ()):
            parts.append(html.escape(format_tensor_line(launch, ordinal, key, ARG_ROLES[ordinal][key], prefix)))
    return "<br>".join(parts)


def hf_mappings_by_ordinal() -> dict[int, list[str]]:
    hf_lookup = {node.node_id: node.lines[0] for node in HF_NODES}
    by_ordinal: dict[int, list[str]] = {}
    for hf_id, ordinal in HF_TO_NN_MAPPING:
        by_ordinal.setdefault(ordinal, []).append(hf_lookup[hf_id])
    return by_ordinal


def table_op_name(launch: dict) -> str:
    op_name = html.escape(launch["op_name"])
    if launch.get("requires_collective"):
        semantic = html.escape(collective_display_name(launch))
        if semantic != op_name:
            return f"`{semantic}`<br>`{op_name}`"
    return f"`{op_name}`"


def render_dot(config: dict, launches: list[dict]) -> str:
    hf_lookup = {node.node_id: node for node in HF_NODES}
    launch_lookup = {launch["ordinal"]: launch for launch in launches}
    lines: list[str] = [
        "digraph Qwen3ComputeLayer0 {",
        '  graph [rankdir=TB, compound=true, splines=polyline, nodesep=0.34, ranksep=0.50,',
        '         fontsize=15, fontname="DejaVu Sans", labelloc=t,',
        '         label="Qwen3-0.6B --fused-kernel=compute layer0, PE=16\\nsolid edges: selected data dependencies only; dashed cross edges: HF<->nncase semantic mapping"];',
        '  node [shape=box, style="rounded,filled", fontname="DejaVu Sans", fontsize=8.5,',
        '        color="#4f5661", margin="0.07,0.045"];',
        '  edge [fontname="DejaVu Sans", fontsize=8, color="#6a7380", arrowsize=0.60];',
        "",
        "  subgraph cluster_hf {",
        '    label="Hugging Face Qwen3 op names";',
        '    color="#8fb6df";',
        '    style="rounded";',
    ]
    for node in HF_NODES:
        lines.append(
            f'    {node.node_id} [label="{dot_label(node.lines)}", fillcolor="#eaf4ff", width=2.95, group="hf"];'
        )
    for idx, row in enumerate(LAYOUT_ROWS):
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
    for ordinal in range(31):
        style = nn_node_style(launch_lookup[ordinal])
        attrs = {
            "label": dot_label(
                nn_label(launch_lookup[ordinal]),
                left=not launch_lookup[ordinal].get("requires_collective"),
            ),
            "group": "nn",
            **style,
        }
        attr_parts = []
        for key, value in attrs.items():
            escaped = value if key == "label" else dot_escape(value)
            attr_parts.append(f'{key}="{escaped}"')
        attr_text = ", ".join(attr_parts)
        lines.append(
            f"    {node_id(ordinal)} [{attr_text}];"
        )
    for idx, row in enumerate(LAYOUT_ROWS):
        if not row.nn:
            lines.append(f'    nb_{idx:02d} [shape=point, style=invis, width=0.02, height=0.02, label=""];')
    lines.extend(["  }", ""])

    for idx, row in enumerate(LAYOUT_ROWS):
        lines.append(f'  sep_{idx:02d} [shape=point, label="", width=0.025, height=0.025, color="#777777", group="sep"];')

    lines.append("")
    for idx, row in enumerate(LAYOUT_ROWS):
        hf_nodes = row.hf or (f"hb_{idx:02d}",)
        nn_nodes = row.nn or (f"nb_{idx:02d}",)
        rank_nodes = [*hf_nodes, f"sep_{idx:02d}", *nn_nodes]
        lines.append("  { rank=same; " + "; ".join(rank_nodes) + "; }")
        ordered = [*hf_nodes, f"sep_{idx:02d}", *nn_nodes]
        for left, right in zip(ordered, ordered[1:]):
            lines.append(f"  {left} -> {right} [style=invis, weight=70];")

    lines.append("")
    for idx in range(len(LAYOUT_ROWS) - 1):
        lines.append(
            f'  sep_{idx:02d} -> sep_{idx + 1:02d} '
            '[style=dashed, color="#777777", arrowhead=none, penwidth=1.15, weight=35];'
        )

    lines.append("\n  // Hugging Face data dependencies.")
    for src, dst in HF_DATA_EDGES:
        attrs = 'color="#527da8", penwidth=1.15'
        if (src, dst) in HF_SKIP_EDGES:
            attrs += ', constraint=false, weight=0.2'
        lines.append(f"  {src} -> {dst} [{attrs}];")

    lines.append("\n  // nncase data dependencies. Auxiliary const/mask/dim_var edges are intentionally omitted.")
    for src, dst in NN_DATA_EDGES:
        attrs = 'color="#a66f24", penwidth=1.15'
        if (src, dst) in NN_SKIP_EDGES:
            attrs += ', constraint=false, weight=0.2'
        lines.append(f"  {node_id(src)} -> {node_id(dst)} [{attrs}];")

    lines.append("\n  // Dashed semantic mapping between Hugging Face ops and lowered nncase ops.")
    for hf_id, ordinal in HF_TO_NN_MAPPING:
        lines.append(
            f'  {hf_id} -> {node_id(ordinal)} '
            '[style=dashed, color="#8a8f98", arrowhead=none, constraint=false, penwidth=0.85];'
        )

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_markdown(config: dict, launches: list[dict]) -> str:
    by_ordinal = hf_mappings_by_ordinal()
    rows = []
    for launch in launches:
        ordinal = launch["ordinal"]
        hf_mapping = "<br>".join(html.escape(name) for name in by_ordinal.get(ordinal, [HF_MEANING[ordinal]]))
        rows.append(
            "| "
            + " | ".join(
                [
                    str(ordinal),
                    table_op_name(launch),
                    hf_mapping,
                    nn_table_io(launch),
                    f"`{HELPERS[ordinal]}`",
                ]
            )
            + " |"
        )

    md_lines = [
        "# Qwen3 `--fused-kernel=compute` Layer0 Graph",
        "",
        "This document maps the current PE=16 compute-mode Qwen3 demo from `input_ids`,",
        "through embedding, to the end of decoder layer 0. The graph is intentionally",
        "browser-width and tall: nncase ops are not merged, solid edges are only selected",
        "data dependencies, and dashed cross-column edges are semantic HF-to-nncase mappings.",
        "Node text emphasizes input -> output shape/SBP transformations instead of raw",
        "argument ordinals.",
        "",
        "Source artifacts:",
        "",
        f"- `{META_PATH.relative_to(ROOT)}`",
        f"- `{LAUNCH_SUMMARY_PATH.relative_to(ROOT)}`",
        f"- `{TRITON_MODULE_PATH.relative_to(ROOT)}`",
        f"- `{CONFIG_PATH.relative_to(ROOT)}`",
        f"- Hugging Face Qwen3 op names: [{HF_QWEN3_SOURCE_URL}]({HF_QWEN3_SOURCE_URL})",
        "",
        "Metadata checks:",
        "",
        "- `pe_count=16`",
        "- `fused_kernel=compute`",
        "- layer0 scope: `main_segment_0_prim` ordinal `0..29`",
        "- ordinal `30` is shown only as the layer1 boundary",
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
        "python docs/triton-backend/generate_qwen3_compute_layer0.py",
        "```",
        "",
        '<img src="qwen3-compute-layer0.svg" alt="Qwen3 compute-mode layer0 graph" style="width: 100%; height: auto;">',
        "",
        "Legend:",
        "",
        "- blue rounded boxes: Hugging Face Qwen3 ops, with input -> output shapes",
        "- orange rounded boxes: ordinary nncase lowered ops",
        "- green rounded boxes: `fusion.cuda.*` compute fused ops",
        "- pink diamonds: `requires_collective=true` nncase launches (`0`, `4`, `14`, `24`)",
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
        "## Compute Fusions Present",
        "",
        "- `fusion.cuda.layer_norm_matmul`",
        "- `fusion.cuda.layer_norm_transpose`",
        "- `fusion.cuda.mul_cos`",
        "- `fusion.cuda.mul_sin`",
        "- `fusion.cuda.rope`",
        "- `fusion.cuda.matmul_silu_matmul_mul_matmul`",
        "",
        "## Boundary Note",
        "",
        "Ordinal `29` calls `device_func_16` for the MLP residual. It also produces the",
        "next layer's pre-norm side output, but the layer0 main output ends at `buffer_169`.",
        "Ordinal `30` is included only to mark the layer1 boundary.",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify generated docs are up to date")
    args = parser.parse_args()

    _meta, config, launches, _launch_summary, _triton_module = load_inputs()
    dot = render_dot(config, launches)
    md = render_markdown(config, launches)
    svg = render_svg(dot)

    ok = True
    ok &= write_or_check(OUT_DOT, dot, args.check)
    ok &= write_or_check(OUT_MD, md, args.check)
    ok &= write_or_check(OUT_SVG, svg, args.check)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
