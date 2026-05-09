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
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import os

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_qwen3_cuda_poc(request):
    try:
        import nncase
        from huggingface_test_runner import (
            CudaAdmissionUnavailable,
            CudaQwenAdmissionRunner,
            download_from_huggingface,
        )
    except (ModuleNotFoundError, RuntimeError) as ex:
        pytest.skip(f"nncase runtime is not importable: {ex}")

    if not nncase.check_target("cuda"):
        pytest.skip("nncase cuda target is not available in this build")

    cfg = """
    [compile_opt]
    shape_bucket_enable = true
    shape_bucket_range_info = { "batch_size"=[1,4], "sequence_length"=[1, 1024] }
    shape_bucket_segments_count = 2
    shape_bucket_fix_var_map = {  }

    [huggingface_options]
    output_logits = true
    output_hidden_states = true
    num_layers = -1
    max_tokens = 3

    [paged_attention_config]
    vectorized_axes = ["HeadDim"]
    lanes = [8]
    sharding_axes = ["NumBlocks"]
    axis_policies = [[0]]
    hierarchy = [1]

    [generator]
    [generator.inputs]
    method = 'text'
    number = 1
    batch = 1

    [generator.inputs.text]
    args = 'tests/importer/huggingface_/prompt.txt'

    [generator.calibs]
    method = 'text'
    number = 1
    batch = 1

    [generator.calibs.text]
    args = 'tests/importer/huggingface_/prompt.txt'

    [target.cpu]
    eval = false
    infer = false

    [target.k510]
    eval = false
    infer = false

    [target.k230]
    eval = false
    infer = false

    [target.xpu]
    eval = false
    infer = false

    [target.cuda]
    eval = false
    infer = true
    similarity_name = 'cosine'

    [target.cuda.mode.noptq]
    enabled = true
    threshold = 0.999

    [target.cuda.mode.ptq]
    enabled = false
    threshold = 0.98

    [target.cuda.target_options]
    Hierarchies = [[1]]
    HierarchyNames = "p"
    UnifiedMemoryArch = false
    MemoryAccessArch = "NUMA"
    """
    runner = CudaQwenAdmissionRunner(request.node.name, overwrite_configs=cfg)

    model_name = "Qwen/Qwen3-0.6B"
    if os.path.exists(os.path.join(os.path.dirname(__file__), model_name)):
        model_file = os.path.join(os.path.dirname(__file__), model_name)
    else:
        model_file = download_from_huggingface(
            AutoModelForCausalLM, AutoTokenizer, model_name, need_save=True)

    try:
        runner.run(model_file)
    except CudaAdmissionUnavailable as ex:
        pytest.xfail(f"CUDA Qwen admission is not ready in this build: {ex}")


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
