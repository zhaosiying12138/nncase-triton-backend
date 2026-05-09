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
#include "runtime_function.h"
#include "nncase/tensor.h"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <nlohmann/json.hpp>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/type.h>
#include <exception>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cuda;

namespace {
using json = nlohmann::json;

std::string span_to_string(std::span<const std::byte> span) {
    return {reinterpret_cast<const char *>(span.data()), span.size_bytes()};
}

result<std::optional<std::string>>
read_optional_text_section(runtime_section_context &context,
                           const char *name) noexcept {
    auto span_r = context.section(name);
    if (span_r.is_ok()) {
        auto span = span_r.unwrap();
        if (!span.empty()) {
            return ok(std::optional<std::string>(span_to_string(span)));
        }
    }

    section_header header;
    auto reader_r = context.seek_section(name, header);
    if (reader_r.is_err()) {
        return ok(std::optional<std::string>());
    }

    std::vector<std::byte> storage(header.body_size);
    if (!storage.empty()) {
        reader_r.unwrap()->read_span(std::span<std::byte>(storage));
    }

    return ok(std::optional<std::string>(
        std::string(reinterpret_cast<const char *>(storage.data()),
                    storage.size())));
}

result<json> parse_json_section(std::string_view text,
                                const char *section) noexcept {
    auto parsed = json::parse(text.begin(), text.end(), nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        std::fprintf(stderr,
                     "nncase cuda runtime: section %s must be a JSON object\n",
                     section);
        return err(std::errc::invalid_argument);
    }

    return ok(std::move(parsed));
}

template <class T>
T json_value_or(const json &value, const char *key, T fallback) {
    auto it = value.find(key);
    if (it == value.end() || it->is_null()) {
        return fallback;
    }

    return it->get<T>();
}

cuda_launch_dim json_dim_or(const json &value, const char *key,
                            cuda_launch_dim fallback) {
    auto it = value.find(key);
    if (it == value.end() || it->is_null()) {
        return fallback;
    }

    if (it->is_array() && it->size() == 3) {
        return cuda_launch_dim{it->at(0).get<uint32_t>(),
                               it->at(1).get<uint32_t>(),
                               it->at(2).get<uint32_t>()};
    }

    if (it->is_object()) {
        return cuda_launch_dim{json_value_or<uint32_t>(*it, "x", fallback.x),
                               json_value_or<uint32_t>(*it, "y", fallback.y),
                               json_value_or<uint32_t>(*it, "z", fallback.z)};
    }

    throw std::invalid_argument("launch dimension must be [x,y,z] or object");
}

const json *select_function_json(const json &meta) {
    auto functions = meta.find("functions");
    if (functions != meta.end()) {
        if (!functions->is_array()) {
            throw std::invalid_argument("functions must be an array");
        }

        if (!functions->empty()) {
            return &functions->at(0);
        }
    }

    if (meta.contains(cuda_func_key_id) || meta.contains(cuda_func_key_name)) {
        return &meta;
    }

    return nullptr;
}

std::vector<std::string> json_string_array_or_empty(const json &value,
                                                    const char *key,
                                                    bool *found = nullptr) {
    auto it = value.find(key);
    if (it == value.end() || it->is_null()) {
        if (found) {
            *found = false;
        }
        return {};
    }

    if (!it->is_array()) {
        throw std::invalid_argument(std::string(key) + " must be an array");
    }

    std::vector<std::string> result;
    result.reserve(it->size());
    for (auto &item : *it) {
        if (!item.is_string()) {
            throw std::invalid_argument(std::string(key) +
                                        " must contain strings");
        }
        result.emplace_back(item.get<std::string>());
    }

    if (found) {
        *found = true;
    }
    return result;
}

class device_allocation {
  public:
    device_allocation() noexcept = default;
    device_allocation(cuda_runtime_module &module, CUdeviceptr ptr) noexcept
        : module_(&module), ptr_(ptr) {}
    device_allocation(const device_allocation &) = delete;
    device_allocation &operator=(const device_allocation &) = delete;
    device_allocation(device_allocation &&other) noexcept
        : module_(other.module_), ptr_(other.ptr_) {
        other.module_ = nullptr;
        other.ptr_ = 0;
    }
    ~device_allocation() {
        if (module_ && ptr_) {
            (void)module_->free_device(ptr_);
        }
    }

    device_allocation &operator=(device_allocation &&other) noexcept {
        if (this != &other) {
            if (module_ && ptr_) {
                (void)module_->free_device(ptr_);
            }
            module_ = other.module_;
            ptr_ = other.ptr_;
            other.module_ = nullptr;
            other.ptr_ = 0;
        }
        return *this;
    }

    CUdeviceptr ptr() const noexcept { return ptr_; }

  private:
    cuda_runtime_module *module_ = nullptr;
    CUdeviceptr ptr_ = 0;
};

std::string trim_copy(std::string_view value) {
    auto begin = value.begin();
    auto end = value.end();
    while (begin != end &&
           std::isspace(static_cast<unsigned char>(*begin)) != 0) {
        ++begin;
    }

    while (begin != end &&
           std::isspace(static_cast<unsigned char>(*(end - 1))) != 0) {
        --end;
    }

    return std::string(begin, end);
}

std::vector<std::string> split_shape_symbols(std::string_view type) {
    auto left = type.find('[');
    auto right = type.rfind(']');
    if (left == std::string_view::npos || right == std::string_view::npos ||
        right <= left) {
        return {};
    }

    std::vector<std::string> dims;
    auto shape = type.substr(left + 1, right - left - 1);
    size_t start = 0;
    while (start <= shape.size()) {
        auto comma = shape.find(',', start);
        auto end = comma == std::string_view::npos ? shape.size() : comma;
        dims.emplace_back(trim_copy(shape.substr(start, end - start)));
        if (comma == std::string_view::npos) {
            break;
        }

        start = comma + 1;
    }

    return dims;
}

bool is_unsigned_integer_token(std::string_view value) {
    if (value.empty()) {
        return false;
    }

    return std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return std::isdigit(ch) != 0;
    });
}
} // namespace

cuda_runtime_function::cuda_runtime_function(runtime_module &rt_module)
    : runtime_function(rt_module) {}

cuda_runtime_module &cuda_runtime_function::module() const noexcept {
    return static_cast<cuda_runtime_module &>(runtime_function::module());
}

result<void> cuda_runtime_function::initialize_core(
    runtime_function_init_context &context) noexcept {
    try_var(func_meta_section,
            read_optional_text_section(context, cuda_function_meta_section));
    try_var(smoke_ptx_section,
            read_optional_text_section(context, cuda_smoke_ptx_section));

    if (!func_meta_section.has_value() && !smoke_ptx_section.has_value()) {
        return ok();
    }

    if (!func_meta_section.has_value()) {
        std::fprintf(stderr,
                     "nncase cuda runtime: %s requires metadata section %s\n",
                     cuda_smoke_ptx_section, cuda_function_meta_section);
        return err(std::errc::no_such_file_or_directory);
    }

    try_var(meta,
            parse_json_section(func_meta_section.value(),
                               cuda_function_meta_section));

    try {
        smoke_kernel_name_ = json_value_or<std::string>(
            meta, cuda_func_key_smoke_kernel_name, "");
        smoke_grid_ =
            json_dim_or(meta, cuda_func_key_smoke_grid, cuda_launch_dim{});
        smoke_block_ =
            json_dim_or(meta, cuda_func_key_smoke_block, cuda_launch_dim{});
        smoke_shared_mem_bytes_ = json_value_or<uint32_t>(
            meta, cuda_func_key_smoke_shared_mem_bytes, 0);

        if (!smoke_kernel_name_.empty()) {
            if (!smoke_ptx_section.has_value()) {
                std::fprintf(stderr,
                             "nncase cuda runtime: %s requires PTX section "
                             "%s\n",
                             cuda_func_key_smoke_kernel_name,
                             cuda_smoke_ptx_section);
                return err(std::errc::no_such_file_or_directory);
            }

            smoke_ptx_ = std::move(smoke_ptx_section.value());
            has_smoke_kernel_ = true;
        }

        if (auto func = select_function_json(meta)) {
            if (!func->is_object()) {
                throw std::invalid_argument("function metadata must be an "
                                            "object");
            }
            if (!func->contains(cuda_func_key_id) ||
                !func->contains(cuda_func_key_name)) {
                throw std::invalid_argument(
                    "function metadata requires id and name");
            }

            function_meta_ = function_meta{};
            function_meta_.id =
                func->at(cuda_func_key_id).get<uint32_t>();
            function_meta_.name =
                func->at(cuda_func_key_name).get<std::string>();
            function_meta_.parameters = json_string_array_or_empty(
                *func, cuda_func_key_parameters,
                &function_meta_.has_parameter_metadata);
            auto parameter_descs =
                func->find(cuda_func_key_parameter_descriptors);
            if (parameter_descs != func->end() && !parameter_descs->is_null()) {
                if (!parameter_descs->is_array()) {
                    throw std::invalid_argument(
                        "parameter_descriptors must be an array");
                }

                function_meta_.parameter_descs.reserve(parameter_descs->size());
                for (auto &desc_json : *parameter_descs) {
                    if (!desc_json.is_object()) {
                        throw std::invalid_argument(
                            "parameter descriptor must be an object");
                    }

                    parameter_meta desc;
                    desc.name =
                        json_value_or<std::string>(desc_json, "name", "");
                    desc.type =
                        json_value_or<std::string>(desc_json, "type", "");
                    function_meta_.parameter_descs.emplace_back(
                        std::move(desc));
                }
            }
            function_meta_.data_pool_size = json_value_or<size_t>(
                *func, cuda_func_key_data_pool_size,
                json_value_or<size_t>(meta, cuda_meta_key_data_pool_size, 0));
            function_meta_.output_pool_size = json_value_or<size_t>(
                *func, cuda_func_key_output_pool_size,
                json_value_or<size_t>(meta, cuda_meta_key_output_pool_size, 0));
            function_meta_.rdata_pool_size = json_value_or<size_t>(
                *func, cuda_func_key_rdata_pool_size,
                json_value_or<size_t>(meta, cuda_meta_key_rdata_pool_size, 0));
            function_meta_.block_local_data_pool_size = json_value_or<size_t>(
                *func, cuda_func_key_block_local_data_pool_size,
                json_value_or<size_t>(
                    meta, cuda_meta_key_block_local_data_pool_size, 0));

            auto launches = func->find(cuda_func_key_launches);
            if (launches != func->end() && !launches->is_null()) {
                if (!launches->is_array()) {
                    throw std::invalid_argument("launches must be an array");
                }

                function_meta_.launches.reserve(launches->size());
                for (size_t i = 0; i < launches->size(); i++) {
                    auto &launch_json = launches->at(i);
                    if (!launch_json.is_object()) {
                        throw std::invalid_argument(
                            "launch entry must be an object");
                    }

                    launch_meta launch;
                    launch.ordinal = json_value_or<uint32_t>(
                        launch_json, "ordinal", static_cast<uint32_t>(i));
                    launch.kind =
                        json_value_or<std::string>(launch_json, "kind", "");
                    launch.op_name = json_value_or<std::string>(
                        launch_json, "op_name", "");
                    launch.arguments =
                        json_string_array_or_empty(launch_json, "arguments");
                    launch.requires_collective = json_value_or<bool>(
                        launch_json, "requires_collective", false);
                    function_meta_.launches.emplace_back(std::move(launch));
                }
            }

            auto read_dim_meta = [](const json &dim_json) -> dim_meta {
                if (!dim_json.is_object()) {
                    throw std::invalid_argument(
                        "dimension descriptor must be an object");
                }

                dim_meta dim;
                dim.kind =
                    json_value_or<std::string>(dim_json, "kind", "");
                auto value = dim_json.find("value");
                if (value != dim_json.end() && !value->is_null()) {
                    dim.value = value->get<size_t>();
                }

                dim.symbol =
                    json_value_or<std::string>(dim_json, "symbol", "");
                dim.expression =
                    json_value_or<std::string>(dim_json, "expression", "");
                return dim;
            };

            auto read_dim_array = [&](const json &parent,
                                      const char *key) -> std::vector<dim_meta> {
                auto it = parent.find(key);
                if (it == parent.end() || it->is_null()) {
                    return {};
                }

                if (!it->is_array()) {
                    throw std::invalid_argument(std::string(key) +
                                                " must be an array");
                }

                std::vector<dim_meta> dims;
                dims.reserve(it->size());
                for (auto &dim_json : *it) {
                    dims.emplace_back(read_dim_meta(dim_json));
                }

                return dims;
            };

            auto read_fixed_size = [&](const json &value) -> size_t {
                if (value.is_number_unsigned() || value.is_number_integer()) {
                    return value.get<size_t>();
                }

                if (value.is_string()) {
                    auto text = value.get<std::string>();
                    if (is_unsigned_integer_token(text)) {
                        return static_cast<size_t>(std::stoull(text));
                    }
                }

                if (value.is_object()) {
                    return read_dim_meta(value).value;
                }

                throw std::invalid_argument("expected fixed byte offset");
            };

            auto read_memory_meta = [&](const json &buffer_json) -> memory_meta {
                auto memory_it = buffer_json.find("memory");
                if (memory_it == buffer_json.end() ||
                    !memory_it->is_object()) {
                    throw std::invalid_argument(
                        "buffer descriptor requires memory");
                }

                memory_meta memory;
                memory.location = json_value_or<std::string>(
                    *memory_it, "location", "");
                auto base = memory_it->find("base_start");
                if (base != memory_it->end() && !base->is_null()) {
                    memory.base_start_bytes = read_fixed_size(*base);
                }

                auto start = memory_it->find("span_start_bytes");
                if (start != memory_it->end() && !start->is_null()) {
                    memory.span_start_bytes = read_fixed_size(*start);
                }

                auto size = memory_it->find("span_size_bytes");
                if (size != memory_it->end() && !size->is_null()) {
                    memory.span_size_bytes = read_fixed_size(*size);
                }

                return memory;
            };

            auto read_buffer_meta = [&](const json &buffer_json) -> buffer_meta {
                if (!buffer_json.is_object()) {
                    throw std::invalid_argument(
                        "buffer descriptor must be an object");
                }

                buffer_meta buffer;
                buffer.name =
                    json_value_or<std::string>(buffer_json, "name", "");
                buffer.shape = read_dim_array(buffer_json, "shape");
                buffer.strides = read_dim_array(buffer_json, "strides");
                buffer.memory = read_memory_meta(buffer_json);
                return buffer;
            };

            auto returns = func->find(cuda_func_key_returns);
            if (returns != func->end() && !returns->is_null()) {
                if (!returns->is_array()) {
                    throw std::invalid_argument("returns must be an array");
                }

                function_meta_.returns.reserve(returns->size());
                for (size_t i = 0; i < returns->size(); i++) {
                    auto &ret_json = returns->at(i);
                    if (!ret_json.is_object()) {
                        throw std::invalid_argument(
                            "return descriptor must be an object");
                    }

                    auto buffer_it = ret_json.find("buffer");
                    if (buffer_it == ret_json.end()) {
                        throw std::invalid_argument(
                            "return descriptor requires buffer");
                    }

                    return_meta ret;
                    ret.index = json_value_or<uint32_t>(
                        ret_json, "index", static_cast<uint32_t>(i));
                    ret.value =
                        json_value_or<std::string>(ret_json, "value", "");
                    ret.buffer = read_buffer_meta(*buffer_it);
                    function_meta_.returns.emplace_back(std::move(ret));
                }
            }

            has_function_meta_ = true;
        }
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "nncase cuda runtime: invalid %s: %s\n",
                     cuda_function_meta_section, ex.what());
        return err(std::errc::invalid_argument);
    } catch (...) {
        std::fprintf(stderr, "nncase cuda runtime: invalid %s\n",
                     cuda_function_meta_section);
        return err(std::errc::invalid_argument);
    }

    return ok();
}

result<size_t> cuda_runtime_function::resolve_dim(
    const dim_meta &dim,
    std::span<const cuda_python_arg> function_args) const noexcept {
    if (dim.kind == "fixed") {
        return ok(dim.value);
    }

    if (!dim.symbol.empty()) {
        for (size_t param = 0;
             param < function_meta_.parameter_descs.size() &&
             param < function_args.size();
             param++) {
            auto symbols =
                split_shape_symbols(function_meta_.parameter_descs[param].type);
            for (size_t axis = 0;
                 axis < symbols.size() &&
                 axis < function_args[param].tensor.shape.size();
                 axis++) {
                if (symbols[axis] == dim.symbol) {
                    return ok(function_args[param].tensor.shape[axis]);
                }
            }
        }
    }

    if (!dim.expression.empty() && is_unsigned_integer_token(dim.expression)) {
        return ok(static_cast<size_t>(std::stoull(dim.expression)));
    }

    std::fprintf(stderr,
                 "nncase cuda runtime: cannot resolve dynamic output "
                 "dimension kind=%s symbol=%s expression=%s for function %u "
                 "(%s)\n",
                 dim.kind.c_str(), dim.symbol.c_str(), dim.expression.c_str(),
                 function_meta_.id, function_meta_.name.c_str());
    return err(std::errc::invalid_argument);
}

result<dims_t> cuda_runtime_function::resolve_shape(
    const buffer_meta &buffer,
    std::span<const cuda_python_arg> function_args) const noexcept {
    dims_t shape;
    try {
        shape.reserve(buffer.shape.size());
        for (auto &dim : buffer.shape) {
            try_var(value, resolve_dim(dim, function_args));
            shape.emplace_back(value);
        }
    } catch (...) {
        return err(std::errc::not_enough_memory);
    }

    return ok(std::move(shape));
}

result<tensor> cuda_runtime_function::create_output_tensor(
    const return_meta &ret, size_t output_id,
    std::span<const cuda_python_arg> function_args) noexcept {
    if (ret.buffer.memory.location != "Output") {
        std::fprintf(stderr,
                     "nncase cuda runtime: CUDA/Triton function %u (%s) "
                     "return %zu uses unsupported memory location %s; only "
                     "Output pool returns are implemented\n",
                     function_meta_.id, function_meta_.name.c_str(), output_id,
                     ret.buffer.memory.location.c_str());
        return err(std::errc::not_supported);
    }

    try_var(output_type, return_type(output_id));
    try_var(ttype, output_type.as<tensor_type>());

    dims_t shape;
    if (!ret.buffer.shape.empty()) {
        try_set(shape, resolve_shape(ret.buffer, function_args));
    } else {
        try_set(shape, ttype->shape().as_fixed());
    }

    auto strides = get_default_strides(shape);
    try_var(output_tensor,
            hrt::create(ttype->dtype(), std::move(shape), strides,
                        hrt::pool_cpu_only));
    try_var(mapped, hrt::map(output_tensor, map_write));
    auto bytes = mapped.buffer().size_bytes();

    size_t pool_bytes = 0;
    try_var(output_pool, module().output_pool(0, &pool_bytes));
    auto offset =
        ret.buffer.memory.base_start_bytes + ret.buffer.memory.span_start_bytes;
    if (offset > pool_bytes || bytes > pool_bytes - offset) {
        std::fprintf(stderr,
                     "nncase cuda runtime: CUDA/Triton function %u (%s) "
                     "return %zu output slice is outside PE0 output pool "
                     "(offset %zu, bytes %zu, pool %zu)\n",
                     function_meta_.id, function_meta_.name.c_str(), output_id,
                     offset, bytes, pool_bytes);
        return err(std::errc::invalid_argument);
    }

    try_(module().copy_device_to_host(
        static_cast<CUdeviceptr>(output_pool + offset), mapped.buffer().data(),
        bytes, nullptr));
    return ok(output_tensor.impl());
}

result<value_t> cuda_runtime_function::create_outputs(
    std::span<const cuda_python_arg> function_args) noexcept {
    auto output_size = return_size();
    if (output_size == 0) {
        return ok(tuple(std::in_place, std::vector<value_t>()));
    }

    if (function_meta_.returns.empty()) {
        std::fprintf(stderr,
                     "nncase cuda runtime: CUDA/Triton function %u (%s) has "
                     "%u returns but no CUDA output metadata; CPU/torch "
                     "fallback is not available\n",
                     function_meta_.id, function_meta_.name.c_str(),
                     output_size);
        return err(std::errc::not_supported);
    }

    std::vector<value_t> outputs(output_size);
    for (auto &ret : function_meta_.returns) {
        if (ret.index >= outputs.size()) {
            std::fprintf(stderr,
                         "nncase cuda runtime: CUDA/Triton function %u (%s) "
                         "return metadata index %u is out of range %zu\n",
                         function_meta_.id, function_meta_.name.c_str(),
                         ret.index, outputs.size());
            return err(std::errc::invalid_argument);
        }

        try_var(output_tensor, create_output_tensor(ret, ret.index,
                                                    function_args));
        outputs[ret.index] = output_tensor;
    }

    for (size_t i = 0; i < outputs.size(); i++) {
        if (outputs[i].empty()) {
            std::fprintf(stderr,
                         "nncase cuda runtime: CUDA/Triton function %u (%s) "
                         "is missing return metadata for output %zu\n",
                         function_meta_.id, function_meta_.name.c_str(), i);
            return err(std::errc::invalid_argument);
        }
    }

    return ok(outputs.size() == 1
                  ? outputs[0]
                  : tuple(std::in_place, std::move(outputs)));
}

result<value_t>
cuda_runtime_function::invoke_core(std::span<value_t> parameters,
                                   value_t return_value) noexcept {
    if (has_function_meta_) {
        auto expected_parameters = function_meta_.has_parameter_metadata
                                       ? function_meta_.parameters.size()
                                       : static_cast<size_t>(parameters_size());
        if (parameters.size() != expected_parameters) {
            std::fprintf(stderr,
                         "nncase cuda runtime: function %u (%s) expects %zu "
                         "parameters, got %zu\n",
                         function_meta_.id, function_meta_.name.c_str(),
                         expected_parameters, parameters.size());
            return err(std::errc::invalid_argument);
        }

        try {
            std::vector<cuda_python_arg> function_args;
            std::vector<device_allocation> device_allocations;
            function_args.reserve(parameters.size());
            device_allocations.reserve(parameters.size());

            for (size_t i = 0; i < parameters.size(); i++) {
                auto tensor_r = parameters[i].as<tensor>();
                if (tensor_r.is_err()) {
                    std::fprintf(stderr,
                                 "nncase cuda runtime: CUDA/Triton function "
                                 "%u (%s) parameter %zu is not a tensor; "
                                 "CPU/torch fallback is not available\n",
                                 function_meta_.id,
                                 function_meta_.name.c_str(), i);
                    return err(std::errc::not_supported);
                }

                auto tensor = tensor_r.unwrap();
                if (!tensor->is_contiguous()) {
                    std::fprintf(stderr,
                                 "nncase cuda runtime: CUDA/Triton function "
                                 "%u (%s) parameter %zu is not contiguous; "
                                 "non-contiguous CUDA marshaling is pending "
                                 "and CPU/torch fallback is not available\n",
                                 function_meta_.id,
                                 function_meta_.name.c_str(), i);
                    return err(std::errc::not_supported);
                }

                auto host_buffer_r = tensor->buffer().as_host();
                if (host_buffer_r.is_err()) {
                    std::fprintf(stderr,
                                 "nncase cuda runtime: CUDA/Triton function "
                                 "%u (%s) parameter %zu is not backed by a "
                                 "host buffer that can be staged to CUDA; "
                                 "CPU/torch fallback is not available\n",
                                 function_meta_.id,
                                 function_meta_.name.c_str(), i);
                    return err(std::errc::not_supported);
                }

                auto host_buffer = host_buffer_r.unwrap();
                try_var(mapped, host_buffer.map(map_read));
                auto bytes = mapped.buffer().size_bytes();
                try_var(device_ptr, module().allocate_device(bytes));
                device_allocation allocation(module(), device_ptr);
                try_(module().copy_host_to_device(
                    device_ptr, mapped.buffer().data(), bytes, nullptr));

                cuda_python_arg arg;
                arg.tensor.data = static_cast<uintptr_t>(device_ptr);
                arg.tensor.shape.assign(tensor->shape().begin(),
                                        tensor->shape().end());
                arg.tensor.strides.assign(tensor->strides().begin(),
                                          tensor->strides().end());
                arg.tensor.bytes = bytes;
                arg.tensor.typecode =
                    static_cast<int32_t>(tensor->dtype()->typecode());
                function_args.emplace_back(std::move(arg));
                device_allocations.emplace_back(std::move(allocation));
            }

            auto pe_count = module().pe_count();
            if (pe_count == 0) {
                std::fprintf(stderr,
                             "nncase cuda runtime: CUDA/Triton function %u "
                             "(%s) has no PE device pools\n",
                             function_meta_.id, function_meta_.name.c_str());
                return err(std::errc::invalid_argument);
            }

            std::vector<uintptr_t> data_pools;
            std::vector<uintptr_t> output_pools;
            std::vector<uintptr_t> rdata_pools;
            data_pools.reserve(pe_count);
            output_pools.reserve(pe_count);
            rdata_pools.reserve(pe_count);
            for (size_t pe = 0; pe < pe_count; pe++) {
                size_t data_pool_bytes = 0;
                size_t output_pool_bytes = 0;
                size_t rdata_pool_bytes = 0;
                try_var(data_pool, module().data_pool(
                                       static_cast<uint32_t>(pe),
                                       &data_pool_bytes));
                try_var(output_pool, module().output_pool(
                                         static_cast<uint32_t>(pe),
                                         &output_pool_bytes));
                try_var(rdata_pool, module().rdata_pool(
                                        static_cast<uint32_t>(pe),
                                        &rdata_pool_bytes));
                if (data_pool_bytes < function_meta_.data_pool_size ||
                    output_pool_bytes < function_meta_.output_pool_size ||
                    rdata_pool_bytes < function_meta_.rdata_pool_size) {
                    std::fprintf(stderr,
                                 "nncase cuda runtime: CUDA/Triton function "
                                 "%u (%s) PE %zu pool is smaller than "
                                 "metadata requires "
                                 "(data %zu/%zu, output %zu/%zu, rdata "
                                 "%zu/%zu)\n",
                                 function_meta_.id,
                                 function_meta_.name.c_str(), pe,
                                 data_pool_bytes,
                                 function_meta_.data_pool_size,
                                 output_pool_bytes,
                                 function_meta_.output_pool_size,
                                 rdata_pool_bytes,
                                 function_meta_.rdata_pool_size);
                    return err(std::errc::invalid_argument);
                }
                data_pools.emplace_back(data_pool);
                output_pools.emplace_back(output_pool);
                rdata_pools.emplace_back(rdata_pool);
            }

            if (pe_count == 1) {
                try_(module().launch_python(
                    function_meta_.id, 0, data_pools[0], output_pools[0],
                    rdata_pools[0], function_args,
                    std::span<const uintptr_t>{},
                    std::span<const uintptr_t>{},
                    std::span<const uintptr_t>{}, nullptr));
            } else {
                try_(module().launch_python(
                    function_meta_.id, 0, data_pools[0], output_pools[0],
                    rdata_pools[0], function_args, data_pools, output_pools,
                    rdata_pools, nullptr));
            }

            return create_outputs(function_args);
        } catch (const std::bad_alloc &) {
            return err(std::errc::not_enough_memory);
        } catch (const std::exception &ex) {
            std::fprintf(stderr,
                         "nncase cuda runtime: CUDA/Triton invoke failed: "
                         "%s\n",
                         ex.what());
            return err(std::errc::invalid_argument);
        } catch (...) {
            std::fprintf(stderr,
                         "nncase cuda runtime: CUDA/Triton invoke failed\n");
            return err(std::errc::invalid_argument);
        }
    }

    if (!has_smoke_kernel_) {
        std::fprintf(stderr,
                     "nncase cuda runtime: function execution requires CUDA/"
                     "Triton codegen sections; CPU/torch fallback is not "
                     "available\n");
        return err(std::errc::not_supported);
    }

    if (!parameters.empty()) {
        std::fprintf(stderr,
                     "nncase cuda runtime: smoke kernel invoke only supports "
                     "no-argument kernels; use native API for explicit data "
                     "movement\n");
        return err(std::errc::not_supported);
    }

    try_(module().launch_smoke_kernel(smoke_ptx_.c_str(),
                                      smoke_kernel_name_.c_str(), smoke_grid_,
                                      smoke_block_, smoke_shared_mem_bytes_,
                                      nullptr, nullptr));

    if (!return_value.empty()) {
        return ok(return_value);
    }

    if (return_size() == 0) {
        return ok(tuple(std::in_place, std::vector<value_t>()));
    }

    return err(std::errc::not_supported);
}
