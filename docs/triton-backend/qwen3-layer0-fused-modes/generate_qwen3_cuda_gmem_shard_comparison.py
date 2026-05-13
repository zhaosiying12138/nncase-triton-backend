#!/usr/bin/env python3
"""Generate Qwen3 CUDA AutoDistributed gmem cap evidence from real metadata."""

from __future__ import annotations

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
REPORT_PATH = ROOT / "docs/triton-backend/qwen3-cuda-auto-distribute-gmem-shard-constraint.md"
MODE_GENERATOR = OUT_DIR / "generate_qwen3_layer0_fused_modes.py"
STEM = "qwen3-gmem-cap-layer0"
FUNCTION_NAME = "main_segment_1_prim"
LAYER0_ORDINALS = tuple(range(35))


def load_mode_generator():
    spec = importlib.util.spec_from_file_location("qwen3_layer0_fused_modes", MODE_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import mode generator: {MODE_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mode_graph = load_mode_generator()
LAYOUT_ROWS = mode_graph.MAIN1_OFF_LAYOUT_ROWS
NN_DATA_EDGES = tuple(
    (src, dst)
    for src, dst in mode_graph.MAIN1_OFF_NN_DATA_EDGES
    if src in LAYER0_ORDINALS and dst in LAYER0_ORDINALS
)
NN_SKIP_EDGES = frozenset(
    (src, dst)
    for src, dst in mode_graph.OFF_SPEC.nn_skip_edges
    if src in LAYER0_ORDINALS and dst in LAYER0_ORDINALS
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    label: str
    root: Path

    @property
    def meta_path(self) -> Path:
        return self.root / "cuda_admission/pe_16/CodeGen/cuda/cuda_meta.json"

    @property
    def solve_path(self) -> Path:
        return self.root / "cuda_admission/pe_16/05_AutoDistributedPass/0_AutoDistributed_cuda/main/Costs/Solve.txt"

    @property
    def token_path(self) -> Path:
        return self.root / "token_compare.txt"


RUNS = (
    RunSpec(
        "nocap",
        "default no-cap",
        ROOT / "tests_output/qwen3_off_default_nocap_final",
    ),
    RunSpec(
        "cap",
        "default cap 589824",
        ROOT / "tests_output/qwen3_off_default_cap_589824_final",
    ),
)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def load_meta(run: RunSpec) -> dict:
    require_file(run.meta_path)
    return json.loads(run.meta_path.read_text(encoding="utf-8-sig"))


def find_function(meta: dict) -> dict:
    for function in meta["functions"]:
        if function["name"] == FUNCTION_NAME:
            return function
    raise KeyError(FUNCTION_NAME)


def parse_solve(path: Path) -> dict[str, str]:
    require_file(path)
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*([A-Za-z][A-Za-z0-9_ ]+?)\s*:\s*(.+?)\s*$", line)
        if match:
            values[match.group(1).strip()] = match.group(2).strip()
    return values


def parse_token_ratio(path: Path) -> str:
    require_file(path)
    ratio = "unknown"
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.search(r"\|\s*([01]\.\d{4})\s*\|", line)
        if match:
            ratio = match.group(1)
    return ratio


def int_value(values: dict[str, str], key: str) -> int:
    return int(values[key].split(",", 1)[0])


def mib(value: int) -> str:
    return f"{value / (1024 * 1024):.2f} MiB"


def buffer_type(function: dict, name: str) -> str | None:
    buffer = function["buffers"].get(name)
    if buffer is None:
        return None
    return buffer.get("distributed_type")


def arg_types(function: dict, launch: dict) -> list[str]:
    types: list[str] = []
    for arg in launch.get("arguments", []):
        dtype = buffer_type(function, arg)
        if dtype is not None:
            types.append(dtype)
    return types


def output_type(function: dict, launch: dict) -> str:
    for arg in reversed(launch.get("arguments", [])):
        dtype = buffer_type(function, arg)
        if dtype is not None:
            return dtype
    return "-"


def signature(function: dict, ordinal: int) -> tuple[str, str, tuple[str, ...]]:
    launches = function["launches"]
    if ordinal >= len(launches):
        return ("<missing>", "<missing>", ())
    launch = launches[ordinal]
    return (
        launch.get("kind", "-"),
        launch.get("op_name", "-"),
        tuple(arg_types(function, launch)),
    )


def output_signature(function: dict, ordinal: int) -> str:
    launches = function["launches"]
    if ordinal >= len(launches):
        return "<missing>"
    return output_type(function, launches[ordinal])


def short_type(dtype: str) -> str:
    if dtype == "-":
        return dtype
    head = dtype.split(", [p:", 1)[0]
    if "], " in head:
        shape, sbp = head.split("], ", 1)
        shape += "]"
        partial = " partial" if "Partial: True" in dtype else ""
        return f"{shape}\\n{sbp}{partial}"
    return dtype.replace(", [p:16], ", "\\n")


def markdown_type(dtype: str) -> str:
    return "`" + dtype.replace("`", "'") + "`"


def op_label(function: dict, ordinal: int) -> str:
    launches = function["launches"]
    if ordinal >= len(launches):
        return "missing"
    launch = launches[ordinal]
    return f"{launch.get('kind', '-')}/{launch.get('op_name', '-')}"


def node_name(run: str, ordinal: int) -> str:
    return f"{run}_{ordinal:02d}"


def placeholder_name(run: str, row_index: int) -> str:
    return f"{run}_blank_{row_index:02d}"


def node_shape(launch: dict) -> str:
    if launch.get("requires_collective") or launch.get("kind") == "collective":
        return "diamond"
    return "box"


def node_fill(run: str, changed: bool) -> str:
    if changed:
        return "#fff1c2" if run == "nocap" else "#d9f2ff"
    return "#fff7e8" if run == "nocap" else "#eaf4ff"


def node_label(function: dict, ordinal: int) -> str:
    launch = function["launches"][ordinal]
    dtype = short_type(output_type(function, launch))
    return html.escape(f"ord{ordinal:02d} {op_label(function, ordinal)}\\n{dtype}")


def write_dot(functions: dict[str, dict], changed: list[int]) -> Path:
    changed_set = set(changed)
    dot_path = OUT_DIR / f"{STEM}.dot"
    lines = [
        "digraph qwen3_gmem_cap_layer0 {",
        "  graph [rankdir=TB, compound=true, splines=polyline, nodesep=0.34, ranksep=0.50,",
        "         bgcolor=\"white\", fontsize=15, fontname=\"DejaVu Sans\", labelloc=t,",
        "         label=\"Qwen3-0.6B CUDA AutoDistributed default no-cap vs cap layer0, PE=16\\nsolid edges: selected layer0 data dependencies; dashed red edges: changed output SBP only\"];",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"DejaVu Sans\", fontsize=8.5,",
        "        color=\"#4f5661\", margin=\"0.07,0.045\"];",
        "  edge [fontname=\"DejaVu Sans\", fontsize=8, color=\"#6a7380\", arrowsize=0.60];",
        "",
    ]
    for run in RUNS:
        function = functions[run.name]
        lines.append(f"  subgraph cluster_{run.name} {{")
        lines.append(f"    label=\"{run.label}: {FUNCTION_NAME} ord0..34\";")
        lines.append(f"    color=\"{'#dfa85e' if run.name == 'nocap' else '#8fb6df'}\";")
        lines.append("    style=\"rounded\";")
        for ordinal in LAYER0_ORDINALS:
            launch = function["launches"][ordinal]
            attrs = {
                "label": node_label(function, ordinal),
                "group": run.name,
                "shape": node_shape(launch),
                "style": "filled" if node_shape(launch) == "diamond" else "rounded,filled",
                "fillcolor": node_fill(run.name, ordinal in changed_set),
                "color": "#dc2626" if ordinal in changed_set else "#4f5661",
                "width": "5.0" if node_shape(launch) == "diamond" else "3.75",
            }
            if node_shape(launch) == "diamond":
                attrs["height"] = "1.35"
            attr_text = ", ".join(f'{key}="{value}"' for key, value in attrs.items())
            lines.append(f"    {node_name(run.name, ordinal)} [{attr_text}];")
        for idx, row in enumerate(LAYOUT_ROWS):
            if not row.nn:
                lines.append(
                    f'    {placeholder_name(run.name, idx)} '
                    '[shape=point, style=invis, width=0.02, height=0.02, label=""];'
                )
        lines.append("  }")
        lines.append("")

    for idx, _row in enumerate(LAYOUT_ROWS):
        lines.append(
            f'  sep_{idx:02d} [shape=point, style=invis, label="", width=0.025, height=0.025, group="sep"];'
        )
    lines.append("")

    for idx, row in enumerate(LAYOUT_ROWS):
        left_nodes = tuple(node.replace("n_", "nocap_") for node in row.nn) or (placeholder_name("nocap", idx),)
        right_nodes = tuple(node.replace("n_", "cap_") for node in row.nn) or (placeholder_name("cap", idx),)
        rank_nodes = [*left_nodes, f"sep_{idx:02d}", *right_nodes]
        lines.append("  { rank=same; " + "; ".join(rank_nodes) + "; }")
        ordered = [*left_nodes, f"sep_{idx:02d}", *right_nodes]
        for left, right in zip(ordered, ordered[1:]):
            lines.append(f"  {left} -> {right} [style=invis, weight=70];")
    lines.append("")

    for idx in range(len(LAYOUT_ROWS) - 1):
        lines.append(f"  sep_{idx:02d} -> sep_{idx + 1:02d} [style=invis, weight=35];")
    lines.append("")

    lines.append("  // Baseline and cap use the same layer0 topology; labels show each run's picked SBP.")
    for run in RUNS:
        for src, dst in NN_DATA_EDGES:
            attrs = 'color="#a66f24", penwidth=1.15' if run.name == "nocap" else 'color="#527da8", penwidth=1.15'
            if (src, dst) in NN_SKIP_EDGES:
                attrs += ", constraint=false, weight=0.2"
            lines.append(f"  {node_name(run.name, src)} -> {node_name(run.name, dst)} [{attrs}];")
    lines.append("")

    lines.append("  // Dashed cross edges are emitted only for changed output distributed types.")
    for ordinal in changed:
        lines.append(
            f"  {node_name('nocap', ordinal)} -> {node_name('cap', ordinal)} "
            '[style=dashed, color="#dc2626", arrowhead=none, constraint=false, penwidth=0.95];'
        )
    lines.append("}")
    dot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dot_path


def render_svg(dot_path: Path) -> Path:
    svg_path = dot_path.with_suffix(".svg")
    completed = subprocess.run(
        ["dot", "-Tsvg", str(dot_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    svg_text = re.sub(
        r'<svg width="[^"]+" height="[^"]+"',
        '<svg width="100%" height="auto"',
        completed.stdout,
        count=1,
    )
    svg_path.write_text(svg_text, encoding="utf-8")
    return svg_path


def write_layer_md(functions: dict[str, dict], changed: list[int], svg_path: Path) -> Path:
    path = OUT_DIR / f"{STEM}.md"
    lines = [
        "# Qwen3 Default No-Cap vs Default Cap Layer0 Shard Comparison",
        "",
        "This file is generated from `cuda_meta.json` for the two complete Qwen3 CUDA PE=16 runs.",
        "",
        f"<img src=\"{svg_path.name}\" alt=\"Qwen3 layer0 no-cap vs cap shard comparison\" style=\"width: 100%; height: auto;\">",
        "",
        f"Changed output distributed-type ordinals in `{FUNCTION_NAME}` ord0..34:",
        "",
        "`" + ", ".join(str(i) for i in changed) + "`",
        "",
        "| Ord | default no-cap op | default no-cap output distributed type | default cap op | default cap output distributed type |",
        "| --- | --- | --- | --- | --- |",
    ]
    for ordinal in LAYER0_ORDINALS:
        nocap = functions["nocap"]["launches"][ordinal]
        cap = functions["cap"]["launches"][ordinal]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(ordinal),
                    f"`{op_label(functions['nocap'], ordinal)}`",
                    markdown_type(output_type(functions["nocap"], nocap)),
                    f"`{op_label(functions['cap'], ordinal)}`",
                    markdown_type(output_type(functions["cap"], cap)),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_report(solves: dict[str, dict[str, str]], token_ratios: dict[str, str], changed: list[int], layer_md: Path, svg_path: Path) -> None:
    live_no = int_value(solves["nocap"], "FinalPickedCudaLiveStepFootprintPeakBytes")
    live_cap = int_value(solves["cap"], "FinalPickedCudaLiveStepFootprintPeakBytes")
    constrained_no = int_value(solves["nocap"], "FinalPickedLocalTensorConstrainedNodePeakBytes")
    constrained_cap = int_value(solves["cap"], "FinalPickedLocalTensorConstrainedNodePeakBytes")
    resident_no = int_value(solves["nocap"], "FinalPickedCudaResidentStepFootprintPeakBytes")
    resident_cap = int_value(solves["cap"], "FinalPickedCudaResidentStepFootprintPeakBytes")
    reduction = live_no / live_cap

    lines = [
        "# Qwen3 CUDA AutoDistributed Gmem Shard Constraint",
        "",
        "This report records the default no-cap vs default finite-cap comparison for Qwen3-0.6B CUDA PE=16 with `NNCASE_CUDA_FUSED_KERNEL=off`.",
        "The evidence is generated from complete Qwen3 runs and from AutoDistributed `Solve.txt` plus CUDA `cuda_meta.json`; runtime allocator peak and rdata dedup are not used as the primary proof.",
        "",
        "## Source Artifacts",
        "",
    ]
    for run in RUNS:
        lines.extend(
            [
                f"- {run.label}: `{run.root.relative_to(ROOT)}`",
                f"  - Solve: `{run.solve_path.relative_to(ROOT)}`",
                f"  - Metadata: `{run.meta_path.relative_to(ROOT)}`",
                f"  - Token compare: `{run.token_path.relative_to(ROOT)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Primary Result",
            "",
            "| Run | CUDA PE gmem cap | Solver status | CUDA op-step constraints | picked live tensor gmem/PE | constrained node peak | resident step footprint (secondary) | token ratio |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            (
                f"| default no-cap | `{solves['nocap']['CudaPeGmemLimitBytes']}` | `{solves['nocap']['Status']}` | "
                f"`{solves['nocap']['CudaOpStepMemory']}` | `{live_no}` ({mib(live_no)}) | "
                f"`{constrained_no}` ({mib(constrained_no)}) | `{resident_no}` ({mib(resident_no)}) | `{token_ratios['nocap']}` |"
            ),
            (
                f"| default cap | `{solves['cap']['CudaPeGmemLimitBytes']}` | `{solves['cap']['Status']}` | "
                f"`{solves['cap']['CudaOpStepMemory']}` | `{live_cap}` ({mib(live_cap)}) | "
                f"`{constrained_cap}` ({mib(constrained_cap)}) | `{resident_cap}` ({mib(resident_cap)}) | `{token_ratios['cap']}` |"
            ),
            "",
            f"The picked live tensor gmem/PE drops from `{live_no}` bytes to `{live_cap}` bytes, a `{reduction:.2f}x` reduction. The resident/static footprint is reported only as secondary context (`{resident_no}` to `{resident_cap}` bytes); it is not used as proof. The primary evidence is the AutoDistributed picked shard/live-tensor footprint, not rdata dedup or runtime allocator behavior.",
            "",
            "## Prefill Layer0 Shard Change",
            "",
            f"The comparison uses `{FUNCTION_NAME}` ord0..34, the existing prefill layer0 slice. Output distributed types differ at these ordinals:",
            "",
            "`" + ", ".join(str(i) for i in changed) + "`",
            "",
            f"- Detailed table: [`{layer_md.relative_to(REPORT_PATH.parent)}`]({layer_md.relative_to(REPORT_PATH.parent)})",
            f"- Graph: [`{svg_path.relative_to(REPORT_PATH.parent)}`]({svg_path.relative_to(REPORT_PATH.parent)})",
            "",
            f"<img src=\"{svg_path.relative_to(REPORT_PATH.parent)}\" alt=\"Qwen3 layer0 no-cap vs cap shard comparison\" style=\"width: 100%; height: auto;\">",
            "",
            "Early layer0 examples:",
            "",
            "- ord0 changes `input_ids` from broadcast local load to sequence-sharded load.",
            "- ord2..4 change the embedding/gather path by adding a sequence-sharded tensor load and selecting a sequence-sharded gather/reshard path under the cap.",
            "- ord5..7 keep the layer0 hidden-state path under the cap before later resharding for matmul-compatible layouts.",
            "",
            "## Verification Commands",
            "",
            "Focused AutoDistributed tests:",
            "",
            "```bash",
            "env DOTNET_ROOT=$HOME/.dotnet/zsy-nncase-dotnet8 PATH=\"$HOME/.dotnet/zsy-nncase-dotnet8:$PATH\" LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \\",
            "  dotnet test src/Nncase.Tests/Nncase.Tests.csproj -c Release --no-restore \\",
            "  --filter \"FullyQualifiedName~UnitTestQwenEmbeddingShardSearch|FullyQualifiedName~UnitTestCudaAutoDistMemoryConstraint\" \\",
            "  --logger \"console;verbosity=minimal\"",
            "```",
            "",
            "Compiler publish used by the Qwen run:",
            "",
            "```bash",
            "env DOTNET_ROOT=$HOME/.dotnet/zsy-nncase-dotnet8 PATH=\"$HOME/.dotnet/zsy-nncase-dotnet8:$PATH\" LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \\",
            "  dotnet publish src/Nncase.Compiler/Nncase.Compiler.csproj -c Release --no-restore --sc false -r linux-x64 -o install -v:minimal",
            "cp -f install/lib/*.so install/",
            "```",
            "",
            "Default cap Qwen run:",
            "",
            "```bash",
            "env NNCASE_CUDA_PE_GMEM_LIMIT_BYTES=589824 TMPDIR=$PWD/tmp/qwen3_cap_589824_final \\",
            "  PYTHONPATH=$PWD/tests:$PWD/install/python:$PWD/install:$PWD/install/lib \\",
            "  LD_LIBRARY_PATH=$PWD/install:$PWD/install/lib:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \\",
            "  NNCASE_COMPILER=$PWD/install/Nncase.Compiler.dll \\",
            "  DOTNET_ROOT=$HOME/.dotnet/zsy-nncase-dotnet8 \\",
            "  PATH=\"$HOME/.dotnet/zsy-nncase-dotnet8:$PATH\" \\",
            "  NNCASE_CUDA_FUSED_KERNEL=off \\",
            "  NNCASE_CUDA_REQUIRED_PE=16 \\",
            "  NNCASE_CUDA_TILE_PE=16 \\",
            "  NNCASE_CUDA_USE_NATIVE_TRITON_KERNELS=1 \\",
            "  NNCASE_CUDA_REQUIRE_TRITON_KERNELS=1 \\",
            "  NNCASE_CUDA_FP32_PARTIALS=1 \\",
            "  python -m pytest -vv -s tests/importer/huggingface_/test_qwen3_cuda.py::test_qwen3_cuda_poc",
            "```",
            "",
            "## Design Notes",
            "",
            "- `NNCASE_CUDA_PE_GMEM_LIMIT_BYTES` is plumbed through CUDA target `MemoryCapacities` and interpreted by AutoDistributed for CUDA only.",
            "- The SAT model adds CUDA op-step memory constraints when the finite cap is present; the cap run above added `8946` such constraints.",
            f"- The selected cap `{solves['cap']['CudaPeGmemLimitBytes']}` is below the default no-cap picked live tensor peak `{live_no}` and forces the picked graph down to `{live_cap}` bytes.",
            "- The CUDA extraction score ignores memory-load/store as a tie-breaker and emphasizes synchronization for both default no-cap and default cap runs. This is a shared CUDA default cost-model change; the finite cap run additionally adds SAT memory constraints and cap-only candidate penalties.",
            "- Under a finite CUDA cap, large block-local rdata constants and dynamic-sequence-split matmul candidates with fully broadcast static RHS are rejected as generic CUDA gmem-cap candidates. This is target-level behavior, not a Qwen-specific name or layer heuristic.",
            "- No finite CUDA cap path enables const rdata deduplication; metadata/runtime pool changes are secondary diagnostics only.",
            "",
            "## Regenerate",
            "",
            "```bash",
            "python docs/triton-backend/qwen3-layer0-fused-modes/generate_qwen3_cuda_gmem_shard_comparison.py",
            "```",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    metas = {run.name: load_meta(run) for run in RUNS}
    functions = {name: find_function(meta) for name, meta in metas.items()}
    solves = {run.name: parse_solve(run.solve_path) for run in RUNS}
    token_ratios = {run.name: parse_token_ratio(run.token_path) for run in RUNS}
    changed = [
        ordinal
        for ordinal in LAYER0_ORDINALS
        if output_signature(functions["nocap"], ordinal) != output_signature(functions["cap"], ordinal)
    ]
    dot_path = write_dot(functions, changed)
    svg_path = render_svg(dot_path)
    layer_md = write_layer_md(functions, changed, svg_path)
    write_report(solves, token_ratios, changed, layer_md, svg_path)
    print(f"wrote {dot_path.relative_to(ROOT)}")
    print(f"wrote {svg_path.relative_to(ROOT)}")
    print(f"wrote {layer_md.relative_to(ROOT)}")
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
