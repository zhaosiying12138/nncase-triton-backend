from posixpath import join
from typing import Sequence
import shutil
import os
import ctypes
import ctypes.util
import json
import subprocess
import numpy as np
from test_runner import *
import io
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
import nncase
from npy2json import convert_npy_to_json
from ml_dtypes import bfloat16
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


class CudaAdmissionUnavailable(RuntimeError):
    pass


def download_from_huggingface(model_api, tokenizer_api, model_name, need_save=False):
    print(f" Downloading \033[32m\033[1m {model_name} \033[0m from huggingface ... ")
    model_dir = os.path.join(os.path.dirname(__file__), "llm", model_name)
    print(f" model_dir: {model_dir}")
    if os.path.exists(model_dir):
        print(f"\033[32m\033[1m {model_name} \033[0m exits in \033[34m\033[5m {model_dir} \033[0m")
        return model_dir
    else:
        hf_home_env = os.getenv("HF_HOME")
        if hf_home_env is None:
            print(
                f"Please set your huggingface cache dir in environment variable\033[31m 10.10.1.11 'export HF_HOME=/compiler/share/huggingface_cache' \033[0m")
            # download the model from huggingface hub
            model_path = snapshot_download(repo_id=model_name)
        else:
            # if the model can't access in huggingface hub, you can download it from other source and put it in the cache dir ($HF_HOME/hub)
            # e.g.: modelscope download --model LLM-Research/Llama-3.2-1B-Instruct --local_dir $HF_HOME/hub/LLM-Research/Llama-3.2-1B-Instruct
            cache_model_dir = os.path.join(hf_home_env, "hub", model_name)
            if (os.path.exists(cache_model_dir)):
                model_path = cache_model_dir
            else:
                model_path = snapshot_download(repo_id=model_name)

    if need_save:
        try:
            model = model_api.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = tokenizer_api.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            raise os.error(
                f"\033[31m Download {model_name} has error. Make sure it's a valid repository. Or check your network!\033[0m")

        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        model_dir = model_path
    print(
        f"\033[32m\033[1m {model_name} \033[0m has been downloaded into \033[34m\033[5m {model_dir} \033[0m")
    return model_dir


def recursive_stack(obj):
    if isinstance(obj, (list, tuple)):
        stacked = [recursive_stack(item) for item in obj]
        if all(isinstance(item, torch.Tensor) for item in stacked):
            return torch.stack(stacked)
        else:
            return stacked
    else:
        # numpy not support bf16 tensor
        if (obj.dtype == torch.bfloat16 or obj.dtype == torch.float16):
            obj = obj.to(torch.float32)
        if (obj.shape[0] != 1):
            return torch.unsqueeze(obj, 0)
        else:
            return obj


def dequantize_weights(model_dir):
    for filename in os.listdir(model_dir):

        filepath = os.path.join(model_dir, filename)
        if filename.endswith(".org.safetensors"):
            new_file = filepath.replace(".org.safetensors", ".safetensors")
            if os.path.exists(new_file):
                continue
            else:
                org_filepath = filepath
                filepath = new_file
        elif filename.endswith(".safetensors"):
            new_file = filepath.replace(".safetensors", ".org.safetensors")
            if os.path.exists(new_file):
                continue
            else:
                org_filepath = new_file
        else:
            continue

        if not os.path.exists(org_filepath):
            os.rename(filepath, org_filepath)

        state_dict = load_file(org_filepath)

        for key in list(state_dict.keys()):
            if key.endswith('weight_scale'):
                scale_tensor = state_dict[key].to(torch.float32)
                weight_key = key.replace('.weight_scale', '.weight')
                if weight_key in state_dict:
                    weight_tensor = state_dict[weight_key]
                    if scale_tensor.numel() == 1 or scale_tensor.shape[0] == weight_tensor.shape[0]:
                        weight_fp32 = weight_tensor.to(torch.float32)
                        scaled_weight = weight_fp32 * scale_tensor
                        state_dict[weight_key] = scaled_weight
                    else:
                        raise RuntimeError(
                            f"\033[31m weight_tensor {weight_key} and scale_tensor {key} shape not match! \033[0m")
                else:
                    print(f"Warning: Corresponding weight {weight_key} not found, skipping.")

        save_file(state_dict, filepath)


def normalize_safetensor(model_dir):
    for filename in os.listdir(model_dir):
        if filename.endswith(".safetensors") and not filename.endswith(".org.safetensors"):
            filepath = os.path.join(model_dir, filename)
            org_filepath = filepath.replace(".safetensors", ".org.safetensors")

            if not os.path.exists(org_filepath):
                os.rename(filepath, org_filepath)

            state_dict = load_file(org_filepath)

            for key in list(state_dict.keys()):
                if key.endswith('_scale'):
                    weight_tensor = state_dict[key]
                    if weight_tensor.dim() == 0:
                        state_dict[key] = weight_tensor.unsqueeze(0)

            save_file(state_dict, filepath)


def restore_weights(model_dir):
    for filename in os.listdir(model_dir):
        if filename.endswith(".org.safetensors"):
            org_path = os.path.join(model_dir, filename)
            restored_path = org_path.replace(".org.safetensors", ".safetensors")
            os.rename(org_path, restored_path)
            print(f"Restored: {restored_path}")


def to_np_type(t: str):
    '''
    string to np.type
    '''
    if t == "float32":
        return np.float32
    elif t == "float16":
        return np.float16
    elif t == "bfloat16":
        return bfloat16
    else:
        return None


def dump_data_to_file(dir_path, file_path, data):
    if not test_utils.in_ci():
        dump_bin_file(os.path.join(dir_path, f'{file_path}.bin'), data)
        dump_txt_file(os.path.join(dir_path, f'{file_path}.txt'), data)
        dump_npy_file(os.path.join(dir_path, f'{file_path}.npy'), data)
        convert_npy_to_json(os.path.join(dir_path, f'{file_path}.npy'), dir_path)


class HuggingfaceTestRunner(TestRunner):
    def __init__(self, case_name, overwrite_configs: str = None):
        super().__init__(case_name, overwrite_configs)
        self.model_type = "huggingface"
        self.num_layers = -1
        self.local_inputs: List[Any] = []

    def decode_token(self, logits: np.ndarray):
        """
            logits: [batch_size, seq_lens, vocab_size]
        """
        new_token = np.argmax(logits[0, -1, :], axis=-1)  # int64
        # Decode HF token
        return (new_token, self.tokenizer.decode(new_token, skip_special_tokens=False))

    def get_result(self, model, token_num, eval_or_infer):
        results = []
        next_token_id = None
        next_token = None
        for idx in range(model.outputs_size):
            res = model.get_output_tensor(idx).to_numpy()

            dump_data_to_file(self.tmp_dir, f'nncase_result_{token_num}_{idx}', res)
            if (self.cfg['huggingface_options']['output_logits']) and idx == 0:
                results.append(res)
                next_token_id, next_token = self.decode_token(res[np.newaxis, ...])
            elif idx == 0:
                logits_to_keep = 0
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep,
                                                                           int) else logits_to_keep
                nncase_logits = self.model.lm_head(torch.tensor(
                    res[np.newaxis, ...][:, slice_indices, :], dtype=self.hf_config.torch_dtype)).detach().to(torch.float32).numpy()
                next_token_id, next_token = self.decode_token(nncase_logits)
            else:
                results.append(res)

        return results, next_token_id, next_token

    def hf_eval(self, model, input_data, token_num):
        for idx, i in enumerate(input_data):
            value = None
            if isinstance(i, nncase._nncase.RefPagedAttentionKVCache):
                value = i.as_ivalue()
            else:
                value = nncase.RuntimeTensor.from_numpy(i)
            model.set_input_tensor(idx, value)

        model.run()
        return self.get_result(model, token_num, "eval")

    def hf_infer(self, model, input_data, token_num):
        for idx, value in enumerate(input_data):
            new_data = None
            if isinstance(value, nncase.PagedAttentionKVCache):
                new_data = nncase.RuntimeTensor.from_object(value)
            else:
                new_data = nncase.RuntimeTensor.from_numpy(value)
            model.set_input_tensor(idx, new_data)
        model.run()
        return self.get_result(model, token_num, "infer")

    def pipeline_run(self, model, infer_or_eval):
        input_ids = self.local_inputs[0]['data'][0].input_ids[0].astype(np.int64)
        loop_data = [input_ids, self.local_inputs[1]['data'][0]]

        token_ids = []
        tokens = []
        results = []
        for i in range(self.cfg['huggingface_options']['max_tokens']):
            result = None
            kv_object = None
            if i == 0:
                kv_object = loop_data[1]
            else:  # update kv cache when decodeing
                current_length = loop_data[0].shape[-1]
                kv_object = self.local_inputs[1]['scheduler'].schedule([0], [current_length])
            if infer_or_eval == "infer":
                result, next_token_id, next_token = self.hf_infer(
                    model, [loop_data[0], kv_object], token_num=i)
            else:
                result, next_token_id, next_token = self.hf_eval(
                    model, [loop_data[0], kv_object], token_num=i)

            if next_token_id == self.tokenizer.eos_token_id:
                print(f"EOS token reached at step {i}")
                break

            token_ids.append(next_token_id)
            tokens.append(next_token)

            loop_data[0] = np.array([next_token_id], dtype=np.int64)

            results.append(result)
        return results, token_ids, tokens

    def from_huggingface(self, model_path):
        pass

    def huggingface_device_map(self):
        return "auto"

    def huggingface_run(self, func, model_file, judge_type):
        if not self.inputs:
            self.parse_model(model_file)

        self.generate_all_data()
        self.write_compile_opt()
        expect_results, expect_token_ids, expect_tokens = self.cpu_infer(model_file)

        targets = self.cfg['target']
        model_content = self.read_model_file(model_file)
        import_options = self.get_import_options()

        compiler = None
        dump_hist = self.cfg['dump_hist']
        for k_target, v_target in targets.items():
            self.tmp_dir = os.path.join(self.case_dir, 'tmp')
            if v_target['eval'] or v_target['infer']:
                compile_options = self.get_compile_options(k_target, model_file, self.tmp_dir)
                compile_options.target_options = self.get_target_options(
                    k_target, v_target.get("target_options", None))
                compiler = nncase.Compiler(compile_options)
                self.import_model(compiler, model_content, import_options)

            for stage in ['eval', 'infer']:
                if v_target[stage]:
                    for k_mode, v_mode in v_target['mode'].items():
                        if v_mode['enabled']:
                            os.makedirs(self.tmp_dir, exist_ok=True)
                            if stage == 'eval':
                                self.local_inputs = [self.inputs[0], self.inputs[1]]
                                evaluator = compiler.create_evaluator(3)
                                actual_results, actual_token_ids, actual_tokens = func(
                                    evaluator, "eval")
                            else:
                                self.local_inputs = [self.inputs[0], self.inputs[2]]
                                compiler.compile()
                                kmodel_path = os.path.join(self.tmp_dir, self.cfg['kmodel_name'])
                                with open(kmodel_path, 'wb') as f:
                                    compiler.gencode(f)
                                sim = nncase.Simulator()
                                with open(kmodel_path, 'rb') as f:
                                    sim.load_model(f)

                                actual_results, actual_token_ids, actual_tokens = func(sim, "infer")

                            target_dir = os.path.join(self.case_dir, stage, k_target)
                            os.makedirs(target_dir, exist_ok=True)
                            mode_dir = os.path.join(target_dir, k_mode)
                            shutil.move(self.tmp_dir, mode_dir)

                            judge, result = self.compare_results(
                                expect_results, actual_results, stage, k_target, "cosine", k_mode, v_mode['threshold'], dump_hist, mode_dir)
                            if not judge:
                                if test_utils.in_ci():
                                    self.clear(self.case_dir)
                                print(f"Fault result in {stage}\n{result}")

                            token_judge, token_result = self.compare_token_result(
                                expect_token_ids, actual_token_ids, stage, k_target, v_mode['threshold'])

                            print(f"gt    :{expect_tokens}\nactual:{actual_tokens}")
                            if not token_judge:
                                if test_utils.in_ci():
                                    self.clear(self.case_dir)
                                assert (token_judge), f"{token_result}"
        if test_utils.in_ci():
            self.clear(self.case_dir)

    def run(self, model_file):
        # if self.cfg['huggingface_options']['pipeline']:
        self.huggingface_run(self.pipeline_run, model_file, "LLM")
        # else:
        #     self.huggingface_run(self.prefill_run, model_file, "cosine")

    def cpu_infer(self, model_file: List[str]):
        self.local_inputs = [self.inputs[0]]
        all_outputs = []
        outputs = []
        tokens_ids = []
        tokens = []
        device = next(self.model.parameters()).device
        for idx, input in enumerate(self.local_inputs):

            tokenizer_data = input['data'][0]
            data = torch.tensor(tokenizer_data.input_ids).to(device)
            atten_mask = torch.tensor(tokenizer_data.attention_mask).to(device)
            hf_past_key_values = None

            for i in range(self.cfg['huggingface_options']['max_tokens']):
                outputs = []
                with torch.no_grad():
                    result = self.model(
                        input_ids=data,
                        attention_mask=atten_mask,
                        past_key_values=hf_past_key_values,
                        return_dict=True,
                        use_cache=True,
                        # output_attentions=False,
                        output_hidden_states=self.generation_config.output_hidden_states,
                    )
                hf_past_key_values = result.past_key_values

                count = 0
                logits = None
                if (self.cfg['huggingface_options']['output_logits']):
                    logits = result.logits.detach().to(torch.float32).cpu().numpy()
                    dump_data_to_file(self.case_dir, f'cpu_result_{i}_{count}', logits[0])
                    outputs.append(logits)
                    count += 1
                else:
                    hidden_states = recursive_stack(result.hidden_states).detach().to(
                        torch.float32).numpy()[-1]
                    dump_data_to_file(self.case_dir, f'cpu_result_{i}_{count}', hidden_states[0])
                    outputs.append(hidden_states[0])
                    count += 1

                    hidden_states = result.hidden_states[-1]
                    logits_to_keep = 0
                    slice_indices = slice(-logits_to_keep,
                                          None) if isinstance(logits_to_keep, int) else logits_to_keep
                    logits = self.model.lm_head(hidden_states)[
                        :, slice_indices, :].detach().to(torch.float32).numpy()
                next_token_id, decoded_token = self.decode_token(logits)
                tokens_ids.append(next_token_id)
                tokens.append(decoded_token)

                # Check for EOS token
                if next_token_id == self.tokenizer.eos_token_id:
                    print(f"EOS token reached at step {i}")
                    break

                data = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

                atten_mask = torch.cat(
                    [atten_mask, torch.ones((1, 1), dtype=atten_mask.dtype, device=device)], dim=1
                )

                if (self.cfg['huggingface_options']['output_hidden_states']):
                    if not test_utils.in_ci():
                        hidden_states = recursive_stack(result.hidden_states).detach().cpu().numpy()
                        hidden_states = np.squeeze(hidden_states, 1)
                        dump_data_to_file(self.case_dir, f'cpu_result_{i}_{count}', hidden_states)
                        outputs.append(hidden_states)
                        count += 1
                all_outputs.append(outputs)
        return all_outputs, tokens_ids, tokens

    def parse_model(self, model_path):
        # if self.cfg['huggingface_options']['tensor_type'] == "bfloat16":
        #     raise RuntimeError(
        #         f"Not support bfloat16 tensor type now (because of ort)! Just 'float16' or 'float32'.")

        config = AutoConfig.from_pretrained(model_path + "/config.json")
        self.hf_config = config

        if self.cfg['huggingface_options']['num_layers'] != -1:
            self.num_layers = self.cfg['huggingface_options']['num_layers']
            config.num_hidden_layers = self.num_layers
        else:
            self.num_layers = config.num_hidden_layers

        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim if hasattr(
            config, "head_dim") else config.hidden_size // config.num_attention_heads

        paged_attention_config = self.cfg['paged_attention_config']

        self.block_size = paged_attention_config['block_size']
        self.num_blocks = paged_attention_config['num_blocks']
        self.max_sessions = paged_attention_config['max_sessions']
        self.max_model_len = (self.block_size * self.num_blocks) // self.max_sessions
        self.kv_type = np.dtype(to_np_type(paged_attention_config['kv_type']))
        self.cache_layout = [getattr(nncase.PagedKVCacheDimKind, item)
                             for item in paged_attention_config['cache_layout']]
        # [ nncase.PagedKVCacheDimKind.it for it in paged_attention_config['cache_layout'] ]
        self.vectorized_axes = [getattr(nncase.PagedKVCacheDimKind, item)
                                for item in paged_attention_config['vectorized_axes']]
        self.lanes = paged_attention_config['lanes']
        self.sharding_axes = [getattr(nncase.PagedKVCacheDimKind, item)
                              for item in paged_attention_config['sharding_axes']]
        self.axis_policies = paged_attention_config['axis_policies']
        self.hierarchy = paged_attention_config['hierarchy']

        self.kv_cache_config = nncase.PagedAttentionConfig(
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.kv_type,
            self.block_size,
            self.cache_layout,
            self.vectorized_axes,
            self.lanes,
            self.sharding_axes,
            self.axis_policies
        )

        self.cfg['huggingface_options']['config'] = self.kv_cache_config

        if hasattr(config, "quantization_config"):
            try:
                qcfg = config.quantization_config
                if isinstance(qcfg, dict):
                    if 'ignored_layers' in qcfg:
                        qcfg['ignore'] = qcfg.pop('ignored_layers')
                        print("[quantization_config] renamed 'ignored_layers' -> 'ignore'")
                else:
                    if hasattr(qcfg, 'ignored_layers') and not hasattr(qcfg, 'ignore'):
                        setattr(qcfg, 'ignore', getattr(qcfg, 'ignored_layers'))
                        try:
                            delattr(qcfg, 'ignored_layers')
                        except Exception:
                            pass
                        print("[quantization_config] attribute 'ignored_layers' renamed to 'ignore'")
            except Exception as e:
                print(f"[quantization_config] rename ignored_layers failed: {e}")
            normalize_safetensor(model_path)
            # dequantize_weights(model_path)
            # delattr(config, "quantization_config")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype="auto", device_map=self.huggingface_device_map(), trust_remote_code=True).eval()
        # restore_weights(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.generation_config = self.model.generation_config
        # self.generation_config.return_dict_in_generate = True # if False, generate only output tokens
        self.generation_config.max_new_tokens = self.cfg['huggingface_options']['max_tokens']
        self.generation_config.do_sample = False
        self.generation_config.temperature = 0.0  # for Stable result
        if (self.cfg['huggingface_options']['output_logits']):
            pass
        else:
            self.generation_config.output_hidden_states = True
        if (self.cfg['huggingface_options']['output_hidden_states']):
            self.generation_config.output_hidden_states = True

        input_dict = {}
        for input_ in self.model.dummy_inputs:
            input_dict["name"] = input_
            input_dict["dtype"] = self.model.dummy_inputs[input_].dtype.__repr__().split('.')[1]
            # TODO: fix dynamic shape
            input_dict['shape'] = [1, "sequence_length"]
            input_dict['model_shape'] = [1, "sequence_length"]
        self.inputs.append(input_dict)
        self.calibs.append(copy.deepcopy(input_dict))

        input_scheduler_eval = nncase._nncase.RefPagedAttentionScheduler(
            self.kv_cache_config, self.num_blocks, self.max_model_len, self.hierarchy)
        calibs_scheduler_eval = nncase._nncase.RefPagedAttentionScheduler(
            self.kv_cache_config, self.num_blocks, self.max_model_len, self.hierarchy)

        self.inputs.append(dict(name='kv_cache_eval', dtype='PagedAttentionKVCache',
                                shape=[], model_shape=[], scheduler=input_scheduler_eval))
        self.calibs.append(dict(name='kv_cache_eval', dtype='PagedAttentionKVCache',
                                shape=[], model_shape=[], scheduler=calibs_scheduler_eval))

        input_scheduler = nncase.PagedAttentionScheduler(
            self.kv_cache_config, self.num_blocks, self.max_model_len, self.hierarchy)
        calibs_scheduler = nncase.PagedAttentionScheduler(
            self.kv_cache_config, self.num_blocks, self.max_model_len, self.hierarchy)

        self.inputs.append(dict(name='kv_cache', dtype='PagedAttentionKVCache',
                                shape=[], model_shape=[], scheduler=input_scheduler))
        self.calibs.append(dict(name='kv_cache', dtype='PagedAttentionKVCache',
                                shape=[], model_shape=[], scheduler=calibs_scheduler))

    def import_model(self, compiler, model_content, import_options):
        compiler.import_huggingface(model_content, import_options)

    def compare_results(self,
                        # [token0:[result0, result1], token1:[result0, result1]]
                        ref_ouputs: List[List[np.ndarray]],
                        test_outputs: List[List[np.ndarray]],
                        stage, target, similarity_name, mode, threshold, dump_hist, dump_dir) -> Tuple[bool, str]:
        i = 0
        judges = []
        result = ''
        for token_idx, (expected_token_result, actual_token_result) in enumerate(zip(ref_ouputs, test_outputs)):
            for idx, (expected, actual) in enumerate(zip(expected_token_result, actual_token_result)):
                expected = expected.astype(np.float32)
                actual = actual.astype(np.float32)
                dump_file = os.path.join(dump_dir, f'nncase_result_{token_idx}_{idx}_hist.csv')
                judge, similarity_info = compare_ndarray(
                    expected, actual, similarity_name, threshold, dump_hist, dump_file)
                result_info = "{0} [ {1} {2} {3} {4} ] Output {5}:".format(
                    'Pass' if judge else 'Fail', stage, target, mode, token_idx, idx)
                result += result_info + similarity_info
                judges.append(judge)

        with open(os.path.join(self.case_dir, 'test_result.txt'), 'a+') as f:
            f.write(result)
        return sum(judges) == len(judges), result

    def compare_token_result(self,
                             ref_ouputs: List,
                             test_outputs: List,
                             stage, target, threshold) -> Tuple[bool, str]:
        assert len(ref_ouputs) == len(test_outputs)
        max_len = len(ref_ouputs)
        match_count = 0
        # for token_idx, (expected_token_result, actual_token_result) in enumerate(zip(ref_ouputs, test_outputs)):
        with open(os.path.join(self.case_dir, 'token_compare.txt'), 'a+', encoding='utf-8') as f:
            f.write(f"# {target} {stage} Token Comparison Results\n")
            f.write("| Index | Huggingface |   nncase    | Match |\n")
            f.write("|-------|-------------|-------------|-------|\n")
            for i in range(max_len):
                expect_token = ref_ouputs[i] if i < len(ref_ouputs) else "N/A"
                actual_token = test_outputs[i] if i < len(test_outputs) else "N/A"
                match_status = "✓" if expect_token == actual_token else "✗"
                match_count += 1 if match_status == "✓" else 0
                expect_str = str(expect_token).replace('|', '\\|').replace('\n', ' ')
                actual_str = str(actual_token).replace('|', '\\|').replace('\n', ' ')

                f.write(f"| {i:5} | {expect_str:11} | {actual_str:11} | {match_status:5} |\n")
            compare_result = float(match_count) / max_len
            f.write(f"|-------|-------------|-------------|-------|\n")
            f.write(f"|       |             |             |{compare_result:1.4f} |\n\n")

            if compare_result > threshold:
                return True, f"All tokens matched!"
            else:
                return False, f"Token match ratio {compare_result:.2f}, {match_count}/{max_len}  below threshold {threshold}"


class CudaQwenAdmissionRunner(HuggingfaceTestRunner):
    admission_file_name = "cuda_pe_admission.json"

    def huggingface_device_map(self):
        return "cpu"

    @staticmethod
    def pe_candidates(sm_count):
        pe = int(sm_count)
        if pe < 1:
            raise ValueError(f"SM count must be positive, got {sm_count}")

        candidates = []
        while True:
            if pe not in candidates:
                candidates.append(pe)
            if pe == 1:
                return candidates
            pe = max(1, pe // 2)

    def cuda_sm_count(self):
        env_value = os.getenv("NNCASE_CUDA_SM_COUNT")
        if env_value:
            try:
                return max(1, int(env_value))
            except ValueError as ex:
                raise CudaAdmissionUnavailable(
                    f"NNCASE_CUDA_SM_COUNT must be an integer, got {env_value!r}") from ex

        cuda_library = "/usr/lib/wsl/lib/libcuda.so.1" if os.path.exists(
            "/usr/lib/wsl/lib/libcuda.so.1") else (ctypes.util.find_library("cuda") or "libcuda.so.1")
        try:
            cuda = ctypes.CDLL(cuda_library)
            if cuda.cuInit(0) == 0:
                device = ctypes.c_int()
                sm_count = ctypes.c_int()
                cu_device_attribute_multiprocessor_count = 16
                if cuda.cuDeviceGet(ctypes.byref(device), 0) == 0 and cuda.cuDeviceGetAttribute(
                        ctypes.byref(sm_count), cu_device_attribute_multiprocessor_count, device) == 0:
                    return max(1, int(sm_count.value))
        except Exception as ex:
            print(f"[cuda admission] failed to query CUDA driver SM count: {ex}")

        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                output = subprocess.check_output(
                    [
                        nvidia_smi,
                        "--query-gpu=multiprocessor_count",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=5,
                )
                sm_counts = [int(line.strip()) for line in output.splitlines() if line.strip()]
                if sm_counts:
                    return max(sm_counts)
            except Exception as ex:
                print(f"[cuda admission] failed to query nvidia-smi SM count: {ex}")

        return 1

    def make_cuda_pe_target_options(self, pe):
        options = nncase.NTTTargetOptions()
        options.Hierarchies = [[int(pe)]]
        options.HierarchyNames = "p"
        options.UnifiedMemoryArch = False
        options.MemoryAccessArch = nncase.MemoryAccessArchitecture.NUMA
        return options

    def _json_safe(self, value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        return str(value)

    def estimate_cuda_candidate(self, compiler):
        for name in ("estimate", "estimate_cost", "get_estimate"):
            if hasattr(compiler, name):
                return {
                    "api": name,
                    "result": self._json_safe(getattr(compiler, name)()),
                }
        return {
            "api": None,
            "result": "estimate API is not exposed by the current Python wrapper",
        }

    def _write_admission_record(self, record):
        path = os.path.join(self.case_dir, self.admission_file_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(record), f, indent=2)

    def _compile_cuda_candidate(self, model_file, model_content, import_options, pe):
        candidate_dir = os.path.join(self.case_dir, "cuda_admission", f"pe_{pe}")
        self.clear(candidate_dir)
        os.makedirs(candidate_dir, exist_ok=True)

        compile_options = self.get_compile_options("cuda", model_file, candidate_dir)
        compile_options.target_options = self.make_cuda_pe_target_options(pe)
        compiler = nncase.Compiler(compile_options)
        self.import_model(compiler, model_content, import_options)
        compiler.compile()

        return compiler, {
            "pe": pe,
            "status": "ok",
            "dump_dir": candidate_dir,
            "target_options": {
                "Hierarchies": [[int(pe)]],
                "HierarchyNames": "p",
                "UnifiedMemoryArch": False,
                "MemoryAccessArch": "NUMA",
            },
            "estimate": self.estimate_cuda_candidate(compiler),
        }

    def admit_cuda_pe(self, model_file, model_content, import_options):
        sm_count = self.cuda_sm_count()
        num_blocks = max(1, int(getattr(self, "num_blocks", sm_count) or sm_count))
        candidate_start = min(sm_count, num_blocks)
        candidates = self.pe_candidates(candidate_start)
        record = {
            "sm_count": sm_count,
            "num_blocks": num_blocks,
            "candidate_start": candidate_start,
            "candidates": [],
            "chosen_pe": None,
        }

        for pe in candidates:
            try:
                compiler, candidate_record = self._compile_cuda_candidate(
                    model_file, model_content, import_options, pe)
                record["candidates"].append(candidate_record)
                record["chosen_pe"] = pe
                self._write_admission_record(record)
                print(f"[cuda admission] selected PE={pe}")
                return pe, compiler
            except Exception as ex:
                record["candidates"].append({
                    "pe": pe,
                    "status": "failed",
                    "error": repr(ex),
                })
                self._write_admission_record(record)
                print(f"[cuda admission] PE={pe} failed: {ex}")

        raise CudaAdmissionUnavailable(
            f"CUDA PE admission failed for candidates {candidates}; "
            f"see {os.path.join(self.case_dir, self.admission_file_name)}")

    def _token_length(self, sample):
        if hasattr(sample, "input_ids"):
            input_ids = sample.input_ids
            if hasattr(input_ids, "shape"):
                return int(input_ids.shape[-1])
            if input_ids:
                return len(input_ids[0])
        if isinstance(sample, np.ndarray):
            return int(sample.shape[-1])
        return None

    def _reschedule_kv_data(self, scheduler, token_entry):
        if not token_entry or "data" not in token_entry:
            return None

        scheduled = []
        for sample in token_entry["data"]:
            token_length = self._token_length(sample)
            if token_length is None:
                return None
            scheduled.append(scheduler.schedule([0], [token_length]))
        return scheduled

    def _replace_scheduler_entry(self, entries, index, name, scheduler, token_entry):
        entry = {
            "name": name,
            "dtype": "PagedAttentionKVCache",
            "shape": [],
            "model_shape": [],
            "scheduler": scheduler,
        }
        data = self._reschedule_kv_data(scheduler, token_entry)
        if data is not None:
            entry["data"] = data

        if len(entries) <= index:
            entries.extend({} for _ in range(index + 1 - len(entries)))
        entries[index] = entry

    def rebuild_paged_attention_schedulers(self, hierarchy):
        self.hierarchy = [int(pe) for pe in hierarchy]

        input_scheduler_eval = nncase._nncase.RefPagedAttentionScheduler(
            self.kv_cache_config, self.num_blocks, self.max_model_len, self.hierarchy)
        calibs_scheduler_eval = nncase._nncase.RefPagedAttentionScheduler(
            self.kv_cache_config, self.num_blocks, self.max_model_len, self.hierarchy)
        input_scheduler = nncase.PagedAttentionScheduler(
            self.kv_cache_config, self.num_blocks, self.max_model_len, self.hierarchy)
        calibs_scheduler = nncase.PagedAttentionScheduler(
            self.kv_cache_config, self.num_blocks, self.max_model_len, self.hierarchy)

        self._replace_scheduler_entry(
            self.inputs, 1, "kv_cache_eval", input_scheduler_eval, self.inputs[0] if self.inputs else None)
        self._replace_scheduler_entry(
            self.calibs, 1, "kv_cache_eval", calibs_scheduler_eval, self.calibs[0] if self.calibs else None)
        self._replace_scheduler_entry(
            self.inputs, 2, "kv_cache", input_scheduler, self.inputs[0] if self.inputs else None)
        self._replace_scheduler_entry(
            self.calibs, 2, "kv_cache", calibs_scheduler, self.calibs[0] if self.calibs else None)

    def huggingface_run(self, func, model_file, judge_type):
        if not self.inputs:
            self.parse_model(model_file)

        self.generate_all_data()
        self.write_compile_opt()
        expect_results, expect_token_ids, expect_tokens = self.cpu_infer(model_file)

        targets = self.cfg["target"]
        cuda_target = targets.get("cuda")
        if not cuda_target or not cuda_target.get("infer"):
            raise CudaAdmissionUnavailable("cuda target is not enabled or not available")

        model_content = self.read_model_file(model_file)
        import_options = self.get_import_options()
        dump_hist = self.cfg["dump_hist"]

        ran = False
        for k_mode, v_mode in cuda_target["mode"].items():
            if not v_mode["enabled"]:
                continue

            ran = True
            chosen_pe, compiler = self.admit_cuda_pe(model_file, model_content, import_options)
            self.rebuild_paged_attention_schedulers([chosen_pe])

            self.tmp_dir = os.path.join(self.case_dir, "tmp")
            self.clear(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
            kmodel_path = os.path.join(self.tmp_dir, self.cfg["kmodel_name"])
            with open(kmodel_path, "wb") as f:
                compiler.gencode(f)

            sim = nncase.Simulator()
            with open(kmodel_path, "rb") as f:
                sim.load_model(f)

            self.local_inputs = [self.inputs[0], self.inputs[2]]
            actual_results, actual_token_ids, actual_tokens = func(sim, "infer")

            target_dir = os.path.join(self.case_dir, "infer", "cuda")
            os.makedirs(target_dir, exist_ok=True)
            mode_dir = os.path.join(target_dir, k_mode)
            if os.path.exists(mode_dir):
                self.clear(mode_dir)
            shutil.move(self.tmp_dir, mode_dir)

            judge, result = self.compare_results(
                expect_results, actual_results, "infer", "cuda", "cosine",
                k_mode, v_mode["threshold"], dump_hist, mode_dir)
            if not judge:
                if test_utils.in_ci():
                    self.clear(self.case_dir)
                print(f"Fault result in cuda infer\n{result}")
                assert judge, f"Fault result in cuda infer\n{result}"

            token_judge, token_result = self.compare_token_result(
                expect_token_ids, actual_token_ids, "infer", "cuda", v_mode["threshold"])

            print(f"gt    :{expect_tokens}\nactual:{actual_tokens}")
            if not token_judge:
                if test_utils.in_ci():
                    self.clear(self.case_dir)
                assert token_judge, token_result

        if not ran:
            raise CudaAdmissionUnavailable("cuda target has no enabled infer mode")

        if test_utils.in_ci():
            self.clear(self.case_dir)
