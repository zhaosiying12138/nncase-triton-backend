/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <nncase/runtime/model.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/runtime_module.h>

BEGIN_NS_NNCASE_RT_MODULE(cuda)

NNCASE_INLINE_VAR constexpr module_kind_t cuda_module_kind =
    to_module_kind("cuda");

NNCASE_INLINE_VAR constexpr const char *cuda_module_meta_section =
    ".cuda.meta";
NNCASE_INLINE_VAR constexpr const char *cuda_triton_module_section =
    ".triton.module";
NNCASE_INLINE_VAR constexpr const char *cuda_triton_source_section =
    ".triton.source";
NNCASE_INLINE_VAR constexpr const char *cuda_function_meta_section =
    ".cuda.func";
NNCASE_INLINE_VAR constexpr const char *cuda_smoke_ptx_section =
    ".cuda.smoke.ptx";

NNCASE_INLINE_VAR constexpr uint32_t cuda_section_schema_version = 1;
NNCASE_INLINE_VAR constexpr uint32_t cuda_runtime_native_api_version = 1;

NNCASE_INLINE_VAR constexpr const char *cuda_meta_key_schema_version =
    "schema_version";
NNCASE_INLINE_VAR constexpr const char *cuda_meta_key_device_id = "device_id";
NNCASE_INLINE_VAR constexpr const char *cuda_meta_key_pe_count = "pe_count";
NNCASE_INLINE_VAR constexpr const char *cuda_meta_key_pool_bytes_per_pe =
    "pool_bytes_per_pe";
NNCASE_INLINE_VAR constexpr const char *cuda_meta_key_ccl_scratch_bytes =
    "ccl_scratch_bytes";
NNCASE_INLINE_VAR constexpr const char *cuda_meta_key_triton_module_name =
    "triton_module_name";
NNCASE_INLINE_VAR constexpr const char *cuda_meta_key_data_pool_size =
    "data_pool_size";
NNCASE_INLINE_VAR constexpr const char *cuda_meta_key_output_pool_size =
    "output_pool_size";
NNCASE_INLINE_VAR constexpr const char *cuda_meta_key_rdata_pool_size =
    "rdata_pool_size";
NNCASE_INLINE_VAR constexpr const char
    *cuda_meta_key_thread_local_rdata_pool_size =
        "thread_local_rdata_pool_size";
NNCASE_INLINE_VAR constexpr const char
    *cuda_meta_key_block_local_rdata_pool_size =
        "block_local_rdata_pool_size";
NNCASE_INLINE_VAR constexpr const char
    *cuda_meta_key_block_local_data_pool_size =
        "block_local_data_pool_size";

NNCASE_INLINE_VAR constexpr const char *cuda_func_key_id = "id";
NNCASE_INLINE_VAR constexpr const char *cuda_func_key_name = "name";
NNCASE_INLINE_VAR constexpr const char *cuda_func_key_parameters =
    "parameters";
NNCASE_INLINE_VAR constexpr const char *cuda_func_key_parameter_descriptors =
    "parameter_descriptors";
NNCASE_INLINE_VAR constexpr const char *cuda_func_key_data_pool_size =
    "data_pool_size";
NNCASE_INLINE_VAR constexpr const char *cuda_func_key_output_pool_size =
    "output_pool_size";
NNCASE_INLINE_VAR constexpr const char *cuda_func_key_rdata_pool_size =
    "rdata_pool_size";
NNCASE_INLINE_VAR constexpr const char
    *cuda_func_key_block_local_data_pool_size =
        "block_local_data_pool_size";
NNCASE_INLINE_VAR constexpr const char *cuda_func_key_launches = "launches";
NNCASE_INLINE_VAR constexpr const char *cuda_func_key_returns = "returns";

NNCASE_INLINE_VAR constexpr const char *cuda_func_key_smoke_kernel_name =
    "smoke_kernel_name";
NNCASE_INLINE_VAR constexpr const char *cuda_func_key_smoke_grid =
    "smoke_grid";
NNCASE_INLINE_VAR constexpr const char *cuda_func_key_smoke_block =
    "smoke_block";
NNCASE_INLINE_VAR constexpr const char *cuda_func_key_smoke_shared_mem_bytes =
    "smoke_shared_mem_bytes";

enum cuda_native_handle_flags : uint32_t {
    cuda_native_handle_context = 0,
    cuda_native_handle_smoke_api = 1,
};

struct cuda_runtime_native_api {
    uint32_t abi_version;
    void *self;
    uintptr_t (*context)(void *self) noexcept;
    uintptr_t (*pe_pool)(void *self, uint32_t pe_index,
                         size_t *bytes) noexcept;
    uintptr_t (*ccl_scratch)(void *self, size_t *bytes) noexcept;
    int (*copy_host_to_device_shard)(void *self, uint32_t pe_index,
                                     const void *src, size_t bytes,
                                     size_t dst_offset,
                                     void *stream) noexcept;
    int (*launch_smoke_kernel)(void *self, const char *ptx,
                               const char *kernel_name, uint32_t grid_x,
                               uint32_t grid_y, uint32_t grid_z,
                               uint32_t block_x, uint32_t block_y,
                               uint32_t block_z, uint32_t shared_mem_bytes,
                               void **args, void *stream) noexcept;
};

NNCASE_API result<std::unique_ptr<runtime_module>>
create_cuda_runtime_module();

END_NS_NNCASE_RT_MODULE
