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

#include <cuda.h>
#include <nncase/runtime/cuda/runtime_module.h>
#include <span>
#include <string>
#include <string_view>
#include <vector>

BEGIN_NS_NNCASE_RT_MODULE(cuda)

struct cuda_launch_dim {
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;
};

struct cuda_python_tensor_arg {
    uintptr_t data = 0;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t bytes = 0;
    int32_t typecode = 0;
};

struct cuda_python_paged_kv_cache_arg {
    bool valid = false;
    int32_t num_seqs = 0;
    int32_t num_tokens = 0;
    cuda_python_tensor_arg context_lens;
    cuda_python_tensor_arg seq_lens;
    cuda_python_tensor_arg block_tables;
    cuda_python_tensor_arg slot_mapping;
};

struct cuda_python_arg {
    cuda_python_tensor_arg tensor;
    cuda_python_paged_kv_cache_arg paged_kv_cache;
};

class cuda_runtime_module final : public runtime_module {
  public:
    cuda_runtime_module() noexcept;
    ~cuda_runtime_module() override;

    result<uintptr_t> native_handle(uint32_t flags) const noexcept override;

    CUcontext context() const noexcept { return context_; }
    size_t pe_count() const noexcept { return pe_pools_.size(); }
    result<uintptr_t> pe_pool(uint32_t pe_index, size_t *bytes) const noexcept;
    result<uintptr_t> data_pool(uint32_t pe_index, size_t *bytes) const noexcept;
    result<uintptr_t>
    output_pool(uint32_t pe_index, size_t *bytes) const noexcept;
    result<uintptr_t>
    rdata_pool(uint32_t pe_index, size_t *bytes) const noexcept;
    result<uintptr_t>
    block_local_rdata_pool(uint32_t pe_index, size_t *bytes) const noexcept;
    result<uintptr_t> ccl_scratch(size_t *bytes) const noexcept;
    result<CUdeviceptr> allocate_device(size_t bytes) noexcept;
    result<void> free_device(CUdeviceptr ptr) noexcept;
    result<void> copy_host_to_device(CUdeviceptr dst, const void *src,
                                     size_t bytes, CUstream stream) noexcept;
    result<void> copy_device_to_host(CUdeviceptr src, void *dst, size_t bytes,
                                     CUstream stream) noexcept;
    result<void> copy_host_to_device_shard(uint32_t pe_index, const void *src,
                                           size_t bytes, size_t dst_offset,
                                           CUstream stream) noexcept;
    result<void> launch_smoke_kernel(const char *ptx, const char *kernel_name,
                                     cuda_launch_dim grid,
                                     cuda_launch_dim block,
                                     uint32_t shared_mem_bytes, void **args,
                                     CUstream stream) noexcept;
    result<void> launch_python(uint32_t function_id, uint32_t pe_id,
                               uintptr_t data_pool, uintptr_t output_pool,
                               uintptr_t rdata_pool,
                               uintptr_t block_local_rdata_pool,
                               uintptr_t ccl_scratch,
                               size_t ccl_scratch_bytes,
                               std::span<const cuda_python_arg> function_args,
                               std::span<const uintptr_t> all_data_pools,
                               std::span<const uintptr_t> all_output_pools,
                               std::span<const uintptr_t> all_rdata_pools,
                               std::span<const uintptr_t>
                                   all_block_local_rdata_pools,
                               CUstream stream) noexcept;

  protected:
    result<void> initialize_before_functions(
        runtime_module_init_context &context) noexcept override;
    result<std::unique_ptr<runtime_function>>
    create_function() noexcept override;

  private:
    struct pe_device_pool {
        CUdeviceptr data = 0;
        CUdeviceptr output = 0;
        CUdeviceptr rdata = 0;
        CUdeviceptr block_local_rdata = 0;
        size_t data_size = 0;
        size_t output_size = 0;
        size_t rdata_size = 0;
        size_t block_local_rdata_size = 0;
        bool owns_rdata = false;
        bool owns_block_local_rdata = false;
    };

    result<void> initialize_python(std::string_view triton_module,
                                   std::string_view triton_source) noexcept;
    result<void> initialize_cuda(int device_id, size_t pe_count,
                                 size_t data_pool_bytes_per_pe,
                                 size_t output_pool_bytes_per_pe,
                                 size_t rdata_pool_bytes_per_pe,
                                 size_t thread_local_rdata_pool_bytes_per_pe,
                                 size_t block_local_rdata_pool_bytes_per_pe,
                                 size_t ccl_scratch_bytes,
                                 std::span<const std::byte> rdata,
                                 std::span<const std::byte> thread_local_rdata,
                                 std::span<const std::byte> block_local_rdata)
        noexcept;
    void release_cuda() noexcept;

    static uintptr_t api_context(void *self) noexcept;
    static uintptr_t api_pe_pool(void *self, uint32_t pe_index,
                                 size_t *bytes) noexcept;
    static uintptr_t api_ccl_scratch(void *self, size_t *bytes) noexcept;
    static int api_copy_host_to_device_shard(void *self, uint32_t pe_index,
                                             const void *src, size_t bytes,
                                             size_t dst_offset,
                                             void *stream) noexcept;
    static int api_launch_smoke_kernel(void *self, const char *ptx,
                                       const char *kernel_name,
                                       uint32_t grid_x, uint32_t grid_y,
                                       uint32_t grid_z, uint32_t block_x,
                                       uint32_t block_y, uint32_t block_z,
                                       uint32_t shared_mem_bytes, void **args,
                                       void *stream) noexcept;

  private:
    CUdevice device_ = 0;
    CUcontext context_ = nullptr;
    bool retained_primary_context_ = false;
    std::vector<pe_device_pool> pe_pools_;
    CUdeviceptr ccl_scratch_ = 0;
    size_t ccl_scratch_size_ = 0;
    std::string triton_module_name_;
    void *python_module_ = nullptr;
    cuda_runtime_native_api native_api_;
};

END_NS_NNCASE_RT_MODULE
