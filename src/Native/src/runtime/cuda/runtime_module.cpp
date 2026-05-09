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
#define PY_SSIZE_T_CLEAN
#include "runtime_module.h"
#include "runtime_function.h"
#include "runtime_rdata_layout.h"
#include <Python.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <new>
#include <nlohmann/json.hpp>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_loader.h>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef NNCASE_TRITON_PYTHON_DEFAULT
#define NNCASE_TRITON_PYTHON_DEFAULT ""
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cuda;

namespace {
using json = nlohmann::json;

std::string trim_copy(std::string value) {
    auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(),
                std::find_if_not(value.begin(), value.end(), is_space));
    value.erase(std::find_if_not(value.rbegin(), value.rend(), is_space).base(),
                value.end());
    return value;
}

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

result<std::vector<std::byte>>
read_optional_binary_section(runtime_section_context &context,
                             const char *name) noexcept {
    auto span_r = context.section(name);
    if (span_r.is_ok()) {
        auto span = span_r.unwrap();
        return ok(std::vector<std::byte>(span.begin(), span.end()));
    }

    section_header header;
    auto reader_r = context.seek_section(name, header);
    if (reader_r.is_err()) {
        return ok(std::vector<std::byte>());
    }

    std::vector<std::byte> storage(header.body_size);
    if (!storage.empty()) {
        reader_r.unwrap()->read_span(std::span<std::byte>(storage));
    }

    return ok(std::move(storage));
}

result<std::string> read_required_text_section(runtime_section_context &context,
                                               const char *name) noexcept {
    try_var(section, read_optional_text_section(context, name));
    if (!section.has_value()) {
        std::fprintf(stderr,
                     "nncase cuda runtime: missing required section %s\n",
                     name);
        return err(std::errc::no_such_file_or_directory);
    }

    return ok(std::move(section.value()));
}

result<void> cuda_check(CUresult status, const char *action) noexcept {
    if (status == CUDA_SUCCESS) {
        return ok();
    }

    const char *name = nullptr;
    const char *message = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &message);
    std::fprintf(stderr, "nncase cuda runtime: %s failed: %s (%s)\n", action,
                 name ? name : "unknown", message ? message : "no message");
    return err(std::errc::io_error);
}

std::string configured_python_executable() {
    if (auto env = std::getenv("NNCASE_TRITON_PYTHON");
        env && std::strlen(env) != 0) {
        return env;
    }

    return NNCASE_TRITON_PYTHON_DEFAULT;
}

void print_python_error(const char *action) noexcept {
    std::fprintf(stderr, "nncase cuda runtime: Python %s failed\n", action);
    if (PyErr_Occurred()) {
        PyErr_Print();
    }
}

bool cuda_verbose_enabled() noexcept {
    const char *value = std::getenv("NNCASE_TRITON_VERBOSE");
    if (!value || std::strlen(value) == 0) {
        value = std::getenv("NNCASE_TRITON_VERBOSES");
    }
    if (!value || std::strlen(value) == 0) {
        value = std::getenv("NNCASE_CUDA_VERBOSE");
    }
    if (!value || std::strlen(value) == 0) {
        return false;
    }

    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text == "1" || text == "true" || text == "yes" || text == "on" ||
           text == "verbose";
}

class py_object_ref {
  public:
    py_object_ref() noexcept = default;
    explicit py_object_ref(PyObject *obj) noexcept : obj_(obj) {}
    py_object_ref(const py_object_ref &) = delete;
    py_object_ref &operator=(const py_object_ref &) = delete;
    py_object_ref(py_object_ref &&other) noexcept : obj_(other.obj_) {
        other.obj_ = nullptr;
    }
    ~py_object_ref() {
        Py_XDECREF(obj_);
    }

    py_object_ref &operator=(py_object_ref &&other) noexcept {
        if (this != &other) {
            Py_XDECREF(obj_);
            obj_ = other.obj_;
            other.obj_ = nullptr;
        }
        return *this;
    }

    PyObject *get() const noexcept { return obj_; }
    PyObject *release() noexcept {
        auto obj = obj_;
        obj_ = nullptr;
        return obj;
    }

  private:
    PyObject *obj_ = nullptr;
};

class py_gil_guard {
  public:
    py_gil_guard() noexcept : state_(PyGILState_Ensure()) {}
    py_gil_guard(const py_gil_guard &) = delete;
    py_gil_guard &operator=(const py_gil_guard &) = delete;
    ~py_gil_guard() { PyGILState_Release(state_); }

  private:
    PyGILState_STATE state_;
};

uint64_t fnv1a64(std::string_view value) noexcept {
    uint64_t hash = 14695981039346656037ull;
    for (auto ch : value) {
        hash ^= static_cast<unsigned char>(ch);
        hash *= 1099511628211ull;
    }

    return hash;
}

std::string hex64(uint64_t value) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << value;
    return stream.str();
}

result<PyObject *> import_cached_triton_source_module(
    std::string_view source) noexcept {
    auto cache_dir_env = std::getenv("NNCASE_TRITON_CACHE_DIR");
    if (!cache_dir_env || std::strlen(cache_dir_env) == 0) {
        return err(std::errc::invalid_argument);
    }

    std::filesystem::path source_path;
    std::string hash_text = hex64(fnv1a64(source));
    try {
        auto cache_dir = std::filesystem::path(cache_dir_env);
        std::filesystem::create_directories(cache_dir);
        source_path = cache_dir / ("nncase_triton_" + hash_text + ".py");
        if (!std::filesystem::exists(source_path)) {
            std::ofstream out(source_path, std::ios::binary);
            if (!out) {
                std::fprintf(stderr,
                             "nncase cuda runtime: cannot create Triton "
                             "source cache file %s\n",
                             source_path.string().c_str());
                return err(std::errc::io_error);
            }

            out.write(source.data(),
                      static_cast<std::streamsize>(source.size()));
        }
    } catch (const std::exception &ex) {
        std::fprintf(stderr,
                     "nncase cuda runtime: cannot prepare Triton source "
                     "cache: %s\n",
                     ex.what());
        return err(std::errc::io_error);
    }

    static std::atomic<uint64_t> module_counter{0};
    auto instance_id = module_counter.fetch_add(1, std::memory_order_relaxed);
    auto module_name = "__nncase_triton_cached_" + hash_text + "_" +
                       std::to_string(instance_id);
    auto source_path_text = source_path.string();

    py_object_ref importlib(PyImport_ImportModule("importlib.util"));
    if (!importlib.get()) {
        print_python_error("import importlib.util for Triton source cache");
        return err(std::errc::invalid_argument);
    }

    py_object_ref spec_from_file_location(
        PyObject_GetAttrString(importlib.get(), "spec_from_file_location"));
    py_object_ref module_from_spec(
        PyObject_GetAttrString(importlib.get(), "module_from_spec"));
    if (!spec_from_file_location.get() || !module_from_spec.get()) {
        print_python_error("resolve importlib.util cache helpers");
        return err(std::errc::invalid_argument);
    }

    py_object_ref spec(PyObject_CallFunction(spec_from_file_location.get(),
                                             "ss", module_name.c_str(),
                                             source_path_text.c_str()));
    if (!spec.get()) {
        print_python_error("create Triton source cache import spec");
        return err(std::errc::invalid_argument);
    }

    py_object_ref module(
        PyObject_CallFunctionObjArgs(module_from_spec.get(), spec.get(),
                                     nullptr));
    if (!module.get()) {
        print_python_error("create Triton source cache module");
        return err(std::errc::invalid_argument);
    }

    auto sys_modules = PyImport_GetModuleDict();
    if (!sys_modules ||
        PyDict_SetItemString(sys_modules, module_name.c_str(), module.get()) <
            0) {
        print_python_error("register Triton source cache module");
        return err(std::errc::invalid_argument);
    }

    py_object_ref loader(PyObject_GetAttrString(spec.get(), "loader"));
    py_object_ref exec_module(
        loader.get() ? PyObject_GetAttrString(loader.get(), "exec_module")
                     : nullptr);
    if (!loader.get() || !exec_module.get()) {
        PyDict_DelItemString(sys_modules, module_name.c_str());
        print_python_error("resolve Triton source cache loader");
        return err(std::errc::invalid_argument);
    }

    py_object_ref exec_result(
        PyObject_CallFunctionObjArgs(exec_module.get(), module.get(),
                                     nullptr));
    if (!exec_result.get()) {
        PyDict_DelItemString(sys_modules, module_name.c_str());
        print_python_error("execute Triton source cache module");
        return err(std::errc::invalid_argument);
    }

    return ok(module.release());
}

PyObject *make_py_uint(uintptr_t value) noexcept {
    return PyLong_FromUnsignedLongLong(
        static_cast<unsigned long long>(value));
}

PyObject *make_py_size_tuple(std::span<const size_t> values) noexcept {
    auto tuple = py_object_ref(PyTuple_New(static_cast<Py_ssize_t>(
        values.size())));
    if (!tuple.get()) {
        return nullptr;
    }

    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(values.size()); i++) {
        auto item = make_py_uint(values[static_cast<size_t>(i)]);
        if (!item) {
            return nullptr;
        }
        PyTuple_SET_ITEM(tuple.get(), i, item);
    }

    return tuple.release();
}

PyObject *make_py_uintptr_tuple(std::span<const uintptr_t> values) noexcept {
    auto tuple = py_object_ref(PyTuple_New(static_cast<Py_ssize_t>(
        values.size())));
    if (!tuple.get()) {
        return nullptr;
    }

    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(values.size()); i++) {
        auto item = make_py_uint(values[static_cast<size_t>(i)]);
        if (!item) {
            return nullptr;
        }
        PyTuple_SET_ITEM(tuple.get(), i, item);
    }

    return tuple.release();
}

PyObject *make_py_tensor_arg(const cuda_python_tensor_arg &arg) noexcept {
    auto tuple = py_object_ref(PyTuple_New(5));
    if (!tuple.get()) {
        return nullptr;
    }

    auto ptr = make_py_uint(arg.data);
    auto shape = make_py_size_tuple(arg.shape);
    auto strides = make_py_size_tuple(arg.strides);
    auto bytes = make_py_uint(arg.bytes);
    auto typecode = PyLong_FromLong(arg.typecode);
    if (!ptr || !shape || !strides || !bytes || !typecode) {
        Py_XDECREF(ptr);
        Py_XDECREF(shape);
        Py_XDECREF(strides);
        Py_XDECREF(bytes);
        Py_XDECREF(typecode);
        return nullptr;
    }

    PyTuple_SET_ITEM(tuple.get(), 0, ptr);
    PyTuple_SET_ITEM(tuple.get(), 1, shape);
    PyTuple_SET_ITEM(tuple.get(), 2, strides);
    PyTuple_SET_ITEM(tuple.get(), 3, bytes);
    PyTuple_SET_ITEM(tuple.get(), 4, typecode);
    return tuple.release();
}

result<void> initialize_python_interpreter() noexcept {
    if (Py_IsInitialized()) {
        return ok();
    }

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.parse_argv = 0;
    config.use_environment = 1;

    auto python_executable = configured_python_executable();
    if (!python_executable.empty()) {
        auto status =
            PyConfig_SetBytesString(&config, &config.program_name,
                                    python_executable.c_str());
        if (PyStatus_Exception(status)) {
            std::fprintf(stderr,
                         "nncase cuda runtime: failed to set Python program "
                         "name: %s\n",
                         status.err_msg ? status.err_msg : "unknown");
            PyConfig_Clear(&config);
            return err(std::errc::invalid_argument);
        }

        status = PyConfig_SetBytesString(&config, &config.executable,
                                         python_executable.c_str());
        if (PyStatus_Exception(status)) {
            std::fprintf(stderr,
                         "nncase cuda runtime: failed to set Python "
                         "executable: %s\n",
                         status.err_msg ? status.err_msg : "unknown");
            PyConfig_Clear(&config);
            return err(std::errc::invalid_argument);
        }
    }

    auto status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) {
        std::fprintf(stderr,
                     "nncase cuda runtime: failed to initialize Python: %s\n",
                     status.err_msg ? status.err_msg : "unknown");
        PyConfig_Clear(&config);
        return err(std::errc::invalid_argument);
    }

    PyConfig_Clear(&config);
    return ok();
}

result<PyObject *> import_or_exec_triton_module(std::string_view module_name,
                                                std::string_view source) {
    auto triton = PyImport_ImportModule("triton");
    if (!triton) {
        print_python_error("import triton");
        return err(std::errc::invalid_argument);
    }
    Py_DECREF(triton);

    std::string py_module_name(module_name);
    if (py_module_name.empty()) {
        py_module_name = "__nncase_triton_kmodel__";
    }

    if (!source.empty()) {
        if (auto cache_dir = std::getenv("NNCASE_TRITON_CACHE_DIR");
            cache_dir && std::strlen(cache_dir) != 0) {
            auto cached_module = import_cached_triton_source_module(source);
            if (cached_module.is_ok()) {
                return cached_module;
            }

            if (PyErr_Occurred()) {
                PyErr_Clear();
            }
            std::fprintf(stderr,
                         "nncase cuda runtime: falling back to in-memory "
                         "Triton source execution\n");
        }

        auto module = PyModule_New(py_module_name.c_str());
        if (!module) {
            print_python_error("create Triton source module");
            return err(std::errc::invalid_argument);
        }

        auto sys_modules = PyImport_GetModuleDict();
        if (!sys_modules ||
            PyDict_SetItemString(sys_modules, py_module_name.c_str(), module) <
                0) {
            Py_DECREF(module);
            print_python_error("register Triton source module");
            return err(std::errc::invalid_argument);
        }

        auto globals = PyModule_GetDict(module);
        auto result = PyRun_StringFlags(std::string(source).c_str(),
                                        Py_file_input, globals, globals,
                                        nullptr);
        if (!result) {
            Py_DECREF(module);
            print_python_error("execute Triton source");
            return err(std::errc::invalid_argument);
        }

        Py_DECREF(result);
        return ok(module);
    }

    auto module = PyImport_ImportModule(py_module_name.c_str());
    if (!module) {
        print_python_error("import Triton module");
        return err(std::errc::invalid_argument);
    }

    return ok(module);
}

template <class T>
T json_value_or(const json &value, const char *key, T fallback) {
    auto it = value.find(key);
    if (it == value.end() || it->is_null()) {
        return fallback;
    }

    return it->get<T>();
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
} // namespace

cuda_runtime_module::cuda_runtime_module() noexcept {
    native_api_ = cuda_runtime_native_api{
        cuda_runtime_native_api_version,
        this,
        &cuda_runtime_module::api_context,
        &cuda_runtime_module::api_pe_pool,
        &cuda_runtime_module::api_ccl_scratch,
        &cuda_runtime_module::api_copy_host_to_device_shard,
        &cuda_runtime_module::api_launch_smoke_kernel,
    };
}

cuda_runtime_module::~cuda_runtime_module() {
    release_cuda();

    if (python_module_ && Py_IsInitialized()) {
        auto gil = PyGILState_Ensure();
        Py_DECREF(reinterpret_cast<PyObject *>(python_module_));
        PyGILState_Release(gil);
    }
}

result<uintptr_t>
cuda_runtime_module::native_handle(uint32_t flags) const noexcept {
    if (flags == cuda_native_handle_context) {
        return ok(reinterpret_cast<uintptr_t>(context_));
    }

    if (flags == cuda_native_handle_smoke_api) {
        return ok(reinterpret_cast<uintptr_t>(&native_api_));
    }

    return err(std::errc::invalid_argument);
}

result<void> cuda_runtime_module::initialize_before_functions(
    runtime_module_init_context &context) noexcept {
    try_var(meta_text, read_required_text_section(context,
                                                  cuda_module_meta_section));
    try_var(meta, parse_json_section(meta_text, cuda_module_meta_section));

    try {
        auto schema_version =
            json_value_or<uint32_t>(meta, cuda_meta_key_schema_version, 0);
        if (schema_version != cuda_section_schema_version) {
            std::fprintf(stderr,
                         "nncase cuda runtime: unsupported %s=%u in %s; "
                         "expected %u\n",
                         cuda_meta_key_schema_version, schema_version,
                         cuda_module_meta_section,
                         cuda_section_schema_version);
            return err(std::errc::invalid_argument);
        }

        auto device_id =
            json_value_or<int>(meta, cuda_meta_key_device_id, 0);
        auto pe_count =
            json_value_or<size_t>(meta, cuda_meta_key_pe_count, 1);
        auto pool_bytes_per_pe = json_value_or<size_t>(
            meta, cuda_meta_key_pool_bytes_per_pe, 0);
        auto data_pool_size = json_value_or<size_t>(
            meta, cuda_meta_key_data_pool_size, pool_bytes_per_pe);
        auto output_pool_size = json_value_or<size_t>(
            meta, cuda_meta_key_output_pool_size, 0);
        auto rdata_pool_size =
            json_value_or<size_t>(meta, cuda_meta_key_rdata_pool_size, 0);
        auto thread_local_rdata_pool_size = json_value_or<size_t>(
            meta, cuda_meta_key_thread_local_rdata_pool_size, 0);
        auto block_local_rdata_pool_size = json_value_or<size_t>(
            meta, cuda_meta_key_block_local_rdata_pool_size, 0);
        auto ccl_scratch_bytes = json_value_or<size_t>(
            meta, cuda_meta_key_ccl_scratch_bytes, 0);
        triton_module_name_ = json_value_or<std::string>(
            meta, cuda_meta_key_triton_module_name, "");

        try_var(triton_module_section,
                read_optional_text_section(context,
                                           cuda_triton_module_section));
        try_var(triton_source_section,
                read_optional_text_section(context,
                                           cuda_triton_source_section));

        auto triton_module_name =
            triton_module_section.has_value()
                ? trim_copy(std::move(triton_module_section.value()))
                : std::string();
        if (!triton_module_name_.empty()) {
            triton_module_name = triton_module_name_;
        }

        auto triton_source = triton_source_section.value_or(std::string());
        if (triton_module_name.empty() && triton_source.empty()) {
            std::fprintf(stderr,
                         "nncase cuda runtime: expected %s or %s for cuda "
                         "module initialization\n",
                         cuda_triton_module_section,
                         cuda_triton_source_section);
            return err(std::errc::no_such_file_or_directory);
        }

        try_(initialize_python(triton_module_name, triton_source));
        try_var(rdata_section, read_optional_binary_section(context, ".rdata"));
        try_var(thread_local_rdata_section,
                read_optional_binary_section(context, ".thread_local_rdata"));
        try_var(block_local_rdata_section,
                read_optional_binary_section(context, ".block_local_rdata"));
        rdata_pool_size = std::max(rdata_pool_size, rdata_section.size());
        try_(initialize_cuda(device_id, pe_count, data_pool_size,
                             output_pool_size, rdata_pool_size,
                             thread_local_rdata_pool_size,
                             block_local_rdata_pool_size, ccl_scratch_bytes,
                             rdata_section, thread_local_rdata_section,
                             block_local_rdata_section));
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "nncase cuda runtime: invalid %s: %s\n",
                     cuda_module_meta_section, ex.what());
        return err(std::errc::invalid_argument);
    } catch (...) {
        std::fprintf(stderr, "nncase cuda runtime: invalid %s\n",
                     cuda_module_meta_section);
        return err(std::errc::invalid_argument);
    }

    return ok();
}

result<std::unique_ptr<runtime_function>>
cuda_runtime_module::create_function() noexcept {
    std::unique_ptr<runtime_function> mod(new (std::nothrow)
                                              cuda_runtime_function(*this));
    if (mod) {
        return ok(std::move(mod));
    }

    return err(std::errc::not_enough_memory);
}

result<void> cuda_runtime_module::initialize_python(
    std::string_view triton_module, std::string_view triton_source) noexcept {
    try_(initialize_python_interpreter());

    auto gil = PyGILState_Ensure();
    auto module_r = import_or_exec_triton_module(triton_module, triton_source);
    if (module_r.is_err()) {
        PyGILState_Release(gil);
        return err(module_r.unwrap_err());
    }

    if (python_module_) {
        Py_DECREF(reinterpret_cast<PyObject *>(python_module_));
    }
    python_module_ = module_r.unwrap();
    PyGILState_Release(gil);
    return ok();
}

result<void> cuda_runtime_module::initialize_cuda(
    int device_id, size_t pe_count, size_t data_pool_bytes_per_pe,
    size_t output_pool_bytes_per_pe, size_t rdata_pool_bytes_per_pe,
    size_t thread_local_rdata_pool_bytes_per_pe,
    size_t block_local_rdata_pool_bytes_per_pe, size_t ccl_scratch_bytes,
    std::span<const std::byte> rdata,
    std::span<const std::byte> thread_local_rdata,
    std::span<const std::byte> block_local_rdata) noexcept {
    CHECK_WITH_ERR(pe_count != 0, std::errc::invalid_argument);
    CHECK_WITH_ERR(pe_count <= std::numeric_limits<uint32_t>::max(),
                   std::errc::invalid_argument);

    std::vector<detail::cuda_rdata_content_entry> thread_local_rdata_entries;
    std::vector<detail::cuda_rdata_content_entry> block_local_rdata_entries;
    if (thread_local_rdata_pool_bytes_per_pe != 0) {
        CHECK_WITH_ERR(!thread_local_rdata.empty(),
                       std::errc::invalid_argument);
        try_set(thread_local_rdata_entries,
                detail::parse_cuda_rdata_multi_content_section(
                    thread_local_rdata, pe_count,
                    thread_local_rdata_pool_bytes_per_pe));
    }

    if (block_local_rdata_pool_bytes_per_pe != 0) {
        CHECK_WITH_ERR(!block_local_rdata.empty(), std::errc::invalid_argument);
        try_var(block_local_rdata_count,
                detail::infer_cuda_rdata_multi_content_count(
                    block_local_rdata, block_local_rdata_pool_bytes_per_pe,
                    pe_count));
        try_set(block_local_rdata_entries,
                detail::parse_cuda_rdata_multi_content_section(
                    block_local_rdata, block_local_rdata_count,
                    block_local_rdata_pool_bytes_per_pe));
    }

    try_(cuda_check(cuInit(0), "cuInit"));
    try_(cuda_check(cuDeviceGet(&device_, device_id), "cuDeviceGet"));
    try_(cuda_check(cuDevicePrimaryCtxRetain(&context_, device_),
                    "cuDevicePrimaryCtxRetain"));
    retained_primary_context_ = true;
    try_(cuda_check(cuCtxSetCurrent(context_), "cuCtxSetCurrent"));

    try {
        pe_pools_.resize(pe_count);
    } catch (...) {
        return err(std::errc::not_enough_memory);
    }

    for (size_t pe_index = 0; pe_index < pe_pools_.size(); pe_index++) {
        auto &pool = pe_pools_[pe_index];
        pool.data_size = data_pool_bytes_per_pe;
        pool.output_size = output_pool_bytes_per_pe;
        try_var(rdata_image, detail::build_cuda_rdata_pool_image(
                                 static_cast<uint32_t>(pe_index),
                                 pe_pools_.size(), rdata_pool_bytes_per_pe,
                                 thread_local_rdata_pool_bytes_per_pe,
                                 block_local_rdata_pool_bytes_per_pe, rdata,
                                 thread_local_rdata_entries,
                                 block_local_rdata_entries));
        pool.rdata_size = rdata_image.size();
        if (pool.data_size != 0) {
            try_(cuda_check(cuMemAlloc(&pool.data, pool.data_size),
                            "cuMemAlloc data pool"));
        }
        if (pool.output_size != 0) {
            try_(cuda_check(cuMemAlloc(&pool.output, pool.output_size),
                            "cuMemAlloc output pool"));
        }
        if (pool.rdata_size != 0) {
            try_(cuda_check(cuMemAlloc(&pool.rdata, pool.rdata_size),
                            "cuMemAlloc rdata pool"));
        }
        if (!rdata_image.empty()) {
            try_(cuda_check(cuMemcpyHtoD(pool.rdata, rdata_image.data(),
                                         rdata_image.size()),
                            "cuMemcpyHtoD rdata pool"));
        }
    }

    ccl_scratch_size_ = ccl_scratch_bytes;
    if (ccl_scratch_size_ != 0) {
        try_(cuda_check(cuMemAlloc(&ccl_scratch_, ccl_scratch_size_),
                        "cuMemAlloc ccl scratch"));
    }

    return ok();
}

void cuda_runtime_module::release_cuda() noexcept {
    if (context_) {
        cuCtxSetCurrent(context_);
    }

    for (auto &pool : pe_pools_) {
        if (pool.data) {
            cuMemFree(pool.data);
            pool.data = 0;
        }
        if (pool.output) {
            cuMemFree(pool.output);
            pool.output = 0;
        }
        if (pool.rdata) {
            cuMemFree(pool.rdata);
            pool.rdata = 0;
        }
        pool.data_size = 0;
        pool.output_size = 0;
        pool.rdata_size = 0;
    }
    pe_pools_.clear();

    if (ccl_scratch_) {
        cuMemFree(ccl_scratch_);
        ccl_scratch_ = 0;
    }
    ccl_scratch_size_ = 0;

    if (retained_primary_context_) {
        cuDevicePrimaryCtxRelease(device_);
        retained_primary_context_ = false;
    }

    context_ = nullptr;
}

result<uintptr_t>
cuda_runtime_module::pe_pool(uint32_t pe_index, size_t *bytes) const noexcept {
    return data_pool(pe_index, bytes);
}

result<uintptr_t>
cuda_runtime_module::data_pool(uint32_t pe_index, size_t *bytes) const noexcept {
    CHECK_WITH_ERR(pe_index < pe_pools_.size(),
                   std::errc::result_out_of_range);
    auto &pool = pe_pools_[pe_index];
    if (bytes) {
        *bytes = pool.data_size;
    }

    return ok(static_cast<uintptr_t>(pool.data));
}

result<uintptr_t> cuda_runtime_module::output_pool(uint32_t pe_index,
                                                   size_t *bytes) const
    noexcept {
    CHECK_WITH_ERR(pe_index < pe_pools_.size(),
                   std::errc::result_out_of_range);
    auto &pool = pe_pools_[pe_index];
    if (bytes) {
        *bytes = pool.output_size;
    }

    return ok(static_cast<uintptr_t>(pool.output));
}

result<uintptr_t> cuda_runtime_module::rdata_pool(uint32_t pe_index,
                                                  size_t *bytes) const
    noexcept {
    CHECK_WITH_ERR(pe_index < pe_pools_.size(),
                   std::errc::result_out_of_range);
    auto &pool = pe_pools_[pe_index];
    if (bytes) {
        *bytes = pool.rdata_size;
    }

    return ok(static_cast<uintptr_t>(pool.rdata));
}

result<uintptr_t>
cuda_runtime_module::ccl_scratch(size_t *bytes) const noexcept {
    if (bytes) {
        *bytes = ccl_scratch_size_;
    }

    return ok(static_cast<uintptr_t>(ccl_scratch_));
}

result<CUdeviceptr> cuda_runtime_module::allocate_device(size_t bytes) noexcept {
    if (bytes == 0) {
        return ok(CUdeviceptr{0});
    }

    try_(cuda_check(cuCtxSetCurrent(context_), "cuCtxSetCurrent"));
    CUdeviceptr ptr = 0;
    try_(cuda_check(cuMemAlloc(&ptr, bytes), "cuMemAlloc tensor"));
    return ok(ptr);
}

result<void> cuda_runtime_module::free_device(CUdeviceptr ptr) noexcept {
    if (!ptr) {
        return ok();
    }

    try_(cuda_check(cuCtxSetCurrent(context_), "cuCtxSetCurrent"));
    return cuda_check(cuMemFree(ptr), "cuMemFree tensor");
}

result<void> cuda_runtime_module::copy_host_to_device(
    CUdeviceptr dst, const void *src, size_t bytes, CUstream stream) noexcept {
    CHECK_WITH_ERR(src || bytes == 0, std::errc::invalid_argument);
    if (bytes == 0) {
        return ok();
    }

    CHECK_WITH_ERR(dst != 0, std::errc::invalid_argument);
    try_(cuda_check(cuCtxSetCurrent(context_), "cuCtxSetCurrent"));
    if (stream) {
        return cuda_check(cuMemcpyHtoDAsync(dst, src, bytes, stream),
                          "cuMemcpyHtoDAsync tensor");
    }

    return cuda_check(cuMemcpyHtoD(dst, src, bytes), "cuMemcpyHtoD tensor");
}

result<void> cuda_runtime_module::copy_device_to_host(
    CUdeviceptr src, void *dst, size_t bytes, CUstream stream) noexcept {
    CHECK_WITH_ERR(dst || bytes == 0, std::errc::invalid_argument);
    if (bytes == 0) {
        return ok();
    }

    CHECK_WITH_ERR(src != 0, std::errc::invalid_argument);
    try_(cuda_check(cuCtxSetCurrent(context_), "cuCtxSetCurrent"));
    if (stream) {
        return cuda_check(cuMemcpyDtoHAsync(dst, src, bytes, stream),
                          "cuMemcpyDtoHAsync tensor");
    }

    return cuda_check(cuMemcpyDtoH(dst, src, bytes), "cuMemcpyDtoH tensor");
}

result<void> cuda_runtime_module::copy_host_to_device_shard(
    uint32_t pe_index, const void *src, size_t bytes, size_t dst_offset,
    CUstream stream) noexcept {
    CHECK_WITH_ERR(src || bytes == 0, std::errc::invalid_argument);
    CHECK_WITH_ERR(pe_index < pe_pools_.size(),
                   std::errc::result_out_of_range);

    auto &pool = pe_pools_[pe_index];
    CHECK_WITH_ERR(dst_offset <= pool.data_size, std::errc::invalid_argument);
    CHECK_WITH_ERR(bytes <= pool.data_size - dst_offset,
                   std::errc::invalid_argument);

    if (bytes == 0) {
        return ok();
    }

    try_(cuda_check(cuCtxSetCurrent(context_), "cuCtxSetCurrent"));
    auto dst = pool.data + dst_offset;
    if (stream) {
        return cuda_check(cuMemcpyHtoDAsync(dst, src, bytes, stream),
                          "cuMemcpyHtoDAsync");
    }

    return cuda_check(cuMemcpyHtoD(dst, src, bytes), "cuMemcpyHtoD");
}

result<void> cuda_runtime_module::launch_smoke_kernel(
    const char *ptx, const char *kernel_name, cuda_launch_dim grid,
    cuda_launch_dim block, uint32_t shared_mem_bytes, void **args,
    CUstream stream) noexcept {
    CHECK_WITH_ERR(ptx && ptx[0] != '\0', std::errc::invalid_argument);
    CHECK_WITH_ERR(kernel_name && kernel_name[0] != '\0',
                   std::errc::invalid_argument);

    try_(cuda_check(cuCtxSetCurrent(context_), "cuCtxSetCurrent"));

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    auto load_r = cuda_check(cuModuleLoadDataEx(&module, ptx, 0, nullptr,
                                                nullptr),
                             "cuModuleLoadDataEx");
    if (load_r.is_err()) {
        return load_r;
    }

    auto result = cuda_check(cuModuleGetFunction(&function, module,
                                                 kernel_name),
                             "cuModuleGetFunction");
    if (result.is_ok()) {
        auto verbose = cuda_verbose_enabled();
        auto begin = std::chrono::steady_clock::now();
        if (verbose) {
            std::fprintf(stdout,
                         "[nncase-cuda] begin smoke_kernel name=%s "
                         "launch<grid=(%u,%u,%u), block=(%u,%u,%u)> "
                         "shared_mem=%u stream=%p\n",
                         kernel_name, grid.x, grid.y, grid.z, block.x,
                         block.y, block.z, shared_mem_bytes,
                         reinterpret_cast<void *>(stream));
            std::fflush(stdout);
        }

        result = cuda_check(cuLaunchKernel(function, grid.x, grid.y, grid.z,
                                           block.x, block.y, block.z,
                                           shared_mem_bytes, stream, args,
                                           nullptr),
                            "cuLaunchKernel");
        if (verbose) {
            auto end = std::chrono::steady_clock::now();
            auto elapsed =
                std::chrono::duration<double, std::milli>(end - begin).count();
            std::fprintf(stdout,
                         "[nncase-cuda] end smoke_kernel name=%s "
                         "elapsed_ms=%.3f\n",
                         kernel_name, elapsed);
            std::fflush(stdout);
        }
    }

    if (result.is_ok() && !stream) {
        result = cuda_check(cuCtxSynchronize(), "cuCtxSynchronize");
    }

    cuModuleUnload(module);
    return result;
}

result<void> cuda_runtime_module::launch_python(
    uint32_t function_id, uint32_t pe_id, uintptr_t data_pool,
    uintptr_t output_pool, uintptr_t rdata_pool,
    std::span<const cuda_python_arg> function_args,
    std::span<const uintptr_t> all_data_pools,
    std::span<const uintptr_t> all_output_pools,
    std::span<const uintptr_t> all_rdata_pools, CUstream stream) noexcept {
    if (!python_module_) {
        std::fprintf(stderr,
                     "nncase cuda runtime: Triton Python module is not "
                     "initialized; CPU/torch fallback is not available\n");
        return err(std::errc::not_supported);
    }

    try_(cuda_check(cuCtxSetCurrent(context_), "cuCtxSetCurrent"));

    py_gil_guard gil;

    try {
        auto module = reinterpret_cast<PyObject *>(python_module_);
        py_object_ref launch(PyObject_GetAttrString(module, "launch"));
        if (!launch.get() || !PyCallable_Check(launch.get())) {
            print_python_error("resolve launch");
            return err(std::errc::invalid_argument);
        }

        auto positional_count =
            static_cast<Py_ssize_t>(5 + function_args.size());
        py_object_ref args(PyTuple_New(positional_count));
        if (!args.get()) {
            print_python_error("allocate launch arguments");
            return err(std::errc::not_enough_memory);
        }

        PyObject *base_args[] = {
            PyLong_FromUnsignedLong(function_id),
            PyLong_FromUnsignedLong(pe_id),
            make_py_uint(data_pool),
            make_py_uint(output_pool),
            make_py_uint(rdata_pool),
        };
        for (Py_ssize_t i = 0; i < 5; i++) {
            if (!base_args[i]) {
                for (Py_ssize_t j = i; j < 5; j++) {
                    Py_XDECREF(base_args[j]);
                }
                print_python_error("create launch base arguments");
                return err(std::errc::not_enough_memory);
            }
            PyTuple_SET_ITEM(args.get(), i, base_args[i]);
        }

        for (size_t i = 0; i < function_args.size(); i++) {
            auto arg = make_py_tensor_arg(function_args[i].tensor);
            if (!arg) {
                print_python_error("create tensor argument");
                return err(std::errc::not_enough_memory);
            }

            PyTuple_SET_ITEM(args.get(), static_cast<Py_ssize_t>(5 + i), arg);
        }

        py_object_ref kwargs(PyDict_New());
        if (!kwargs.get()) {
            print_python_error("allocate launch keyword arguments");
            return err(std::errc::not_enough_memory);
        }

        py_object_ref stream_arg;
        if (stream) {
            stream_arg = py_object_ref(make_py_uint(
                reinterpret_cast<uintptr_t>(stream)));
        } else {
            Py_INCREF(Py_None);
            stream_arg = py_object_ref(Py_None);
        }

        if (!stream_arg.get() ||
            PyDict_SetItemString(kwargs.get(), "stream", stream_arg.get()) <
                0) {
            print_python_error("set launch stream");
            return err(std::errc::not_enough_memory);
        }

        if (!all_data_pools.empty() || !all_output_pools.empty() ||
            !all_rdata_pools.empty()) {
            auto data_pools = py_object_ref(make_py_uintptr_tuple(
                all_data_pools));
            auto output_pools = py_object_ref(make_py_uintptr_tuple(
                all_output_pools));
            auto rdata_pools = py_object_ref(make_py_uintptr_tuple(
                all_rdata_pools));
            if (!data_pools.get() || !output_pools.get() ||
                !rdata_pools.get() ||
                PyDict_SetItemString(kwargs.get(), "all_data_pools",
                                     data_pools.get()) < 0 ||
                PyDict_SetItemString(kwargs.get(), "all_output_pools",
                                     output_pools.get()) < 0 ||
                PyDict_SetItemString(kwargs.get(), "all_rdata_pools",
                                     rdata_pools.get()) < 0) {
                print_python_error("set launch PE pool lists");
                return err(std::errc::not_enough_memory);
            }
        }

        auto verbose = cuda_verbose_enabled();
        auto begin = std::chrono::steady_clock::now();
        if (verbose) {
            auto pe_count = !all_data_pools.empty() ? all_data_pools.size() : 1;
            std::fprintf(stdout,
                         "[nncase-cuda] begin python_launch function_id=%u "
                         "pe_id=%u pe_count=%zu arg_count=%zu stream=%p\n",
                         function_id, pe_id, pe_count, function_args.size(),
                         reinterpret_cast<void *>(stream));
            std::fflush(stdout);
        }

        py_object_ref result(PyObject_Call(launch.get(), args.get(),
                                           kwargs.get()));
        if (verbose) {
            auto end = std::chrono::steady_clock::now();
            auto elapsed =
                std::chrono::duration<double, std::milli>(end - begin).count();
            std::fprintf(stdout,
                         "[nncase-cuda] end python_launch function_id=%u "
                         "pe_id=%u elapsed_ms=%.3f\n",
                         function_id, pe_id, elapsed);
            std::fflush(stdout);
        }
        if (!result.get()) {
            print_python_error("call launch");
            return err(std::errc::invalid_argument);
        }
    } catch (const std::exception &ex) {
        std::fprintf(stderr,
                     "nncase cuda runtime: Python launch bridge failed: %s\n",
                     ex.what());
        return err(std::errc::invalid_argument);
    } catch (...) {
        std::fprintf(stderr,
                     "nncase cuda runtime: Python launch bridge failed\n");
        return err(std::errc::invalid_argument);
    }

    if (!stream) {
        try_(cuda_check(cuCtxSynchronize(),
                        "cuCtxSynchronize after Triton launch"));
    }

    return ok();
}

uintptr_t cuda_runtime_module::api_context(void *self) noexcept {
    auto module = reinterpret_cast<cuda_runtime_module *>(self);
    return reinterpret_cast<uintptr_t>(module->context());
}

uintptr_t cuda_runtime_module::api_pe_pool(void *self, uint32_t pe_index,
                                           size_t *bytes) noexcept {
    auto module = reinterpret_cast<cuda_runtime_module *>(self);
    auto result = module->pe_pool(pe_index, bytes);
    return result.is_ok() ? result.unwrap() : 0;
}

uintptr_t cuda_runtime_module::api_ccl_scratch(void *self,
                                               size_t *bytes) noexcept {
    auto module = reinterpret_cast<cuda_runtime_module *>(self);
    auto result = module->ccl_scratch(bytes);
    return result.is_ok() ? result.unwrap() : 0;
}

int cuda_runtime_module::api_copy_host_to_device_shard(
    void *self, uint32_t pe_index, const void *src, size_t bytes,
    size_t dst_offset, void *stream) noexcept {
    auto module = reinterpret_cast<cuda_runtime_module *>(self);
    auto result = module->copy_host_to_device_shard(
        pe_index, src, bytes, dst_offset, reinterpret_cast<CUstream>(stream));
    return result.is_ok() ? 0 : result.unwrap_err().value();
}

int cuda_runtime_module::api_launch_smoke_kernel(
    void *self, const char *ptx, const char *kernel_name, uint32_t grid_x,
    uint32_t grid_y, uint32_t grid_z, uint32_t block_x, uint32_t block_y,
    uint32_t block_z, uint32_t shared_mem_bytes, void **args,
    void *stream) noexcept {
    auto module = reinterpret_cast<cuda_runtime_module *>(self);
    auto result = module->launch_smoke_kernel(
        ptx, kernel_name, cuda_launch_dim{grid_x, grid_y, grid_z},
        cuda_launch_dim{block_x, block_y, block_z}, shared_mem_bytes, args,
        reinterpret_cast<CUstream>(stream));
    return result.is_ok() ? 0 : result.unwrap_err().value();
}

result<std::unique_ptr<runtime_module>>
cuda::create_cuda_runtime_module() {
    std::unique_ptr<runtime_module> mod(new (std::nothrow)
                                            cuda_runtime_module());
    if (mod) {
        return ok(std::move(mod));
    }

    return err(std::errc::not_enough_memory);
}

extern "C" NNCASE_API void RUNTIME_MODULE_ACTIVATOR_NAME(
    result<std::unique_ptr<runtime_module>> &result) {
    result = cuda::create_cuda_runtime_module();
}
