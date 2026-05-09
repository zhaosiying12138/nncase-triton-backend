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

#include "runtime_module.h"
#include <cstddef>
#include <cstdint>
#include <nncase/tensor.h>
#include <nncase/runtime/runtime_function.h>
#include <span>
#include <string>
#include <vector>

BEGIN_NS_NNCASE_RT_MODULE(cuda)

class cuda_runtime_function final : public runtime_function {
  public:
    explicit cuda_runtime_function(runtime_module &rt_module);
    ~cuda_runtime_function() override = default;

    cuda_runtime_module &module() const noexcept;

  protected:
    result<void>
    initialize_core(runtime_function_init_context &context) noexcept override;
    result<value_t> invoke_core(std::span<value_t> parameters,
                                value_t return_value) noexcept override;

  private:
    struct launch_meta {
        uint32_t ordinal = 0;
        std::string kind;
        std::string op_name;
        std::vector<std::string> arguments;
        bool requires_collective = false;
    };

    struct dim_meta {
        std::string kind;
        size_t value = 0;
        std::string symbol;
        std::string expression;
    };

    struct memory_meta {
        std::string location;
        size_t base_start_bytes = 0;
        size_t span_start_bytes = 0;
        size_t span_size_bytes = 0;
    };

    struct buffer_meta {
        std::string name;
        std::vector<dim_meta> shape;
        std::vector<dim_meta> strides;
        memory_meta memory;
    };

    struct return_meta {
        uint32_t index = 0;
        std::string value;
        buffer_meta buffer;
    };

    struct parameter_meta {
        std::string name;
        std::string type;
    };

    struct function_meta {
        uint32_t id = 0;
        std::string name;
        std::vector<std::string> parameters;
        std::vector<parameter_meta> parameter_descs;
        size_t data_pool_size = 0;
        size_t output_pool_size = 0;
        size_t rdata_pool_size = 0;
        size_t block_local_data_pool_size = 0;
        std::vector<launch_meta> launches;
        std::vector<return_meta> returns;
        bool has_parameter_metadata = false;
    };

    result<value_t> create_outputs(
        std::span<const cuda_python_arg> function_args) noexcept;
    result<tensor>
    create_output_tensor(const return_meta &ret, size_t output_id,
                         std::span<const cuda_python_arg> function_args) noexcept;
    result<dims_t> resolve_shape(
        const buffer_meta &buffer,
        std::span<const cuda_python_arg> function_args) const noexcept;
    result<size_t>
    resolve_dim(const dim_meta &dim,
                std::span<const cuda_python_arg> function_args) const noexcept;

    std::string smoke_ptx_;
    std::string smoke_kernel_name_;
    cuda_launch_dim smoke_grid_;
    cuda_launch_dim smoke_block_;
    uint32_t smoke_shared_mem_bytes_ = 0;
    bool has_smoke_kernel_ = false;
    function_meta function_meta_;
    bool has_function_meta_ = false;
};

END_NS_NNCASE_RT_MODULE
