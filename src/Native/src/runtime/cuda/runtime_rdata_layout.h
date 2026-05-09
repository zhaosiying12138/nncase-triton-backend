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
#include <nncase/runtime/result.h>
#include <span>
#include <vector>

BEGIN_NS_NNCASE_RT_MODULE(cuda)

namespace detail {

struct cuda_rdata_content_entry {
    size_t offset = 0;
    std::span<const std::byte> bytes;
};

result<std::vector<cuda_rdata_content_entry>>
parse_cuda_rdata_multi_content_section(std::span<const std::byte> section,
                                       size_t content_count,
                                       size_t max_content_size) noexcept;

result<size_t> infer_cuda_rdata_multi_content_count(
    std::span<const std::byte> section, size_t max_content_size,
    size_t max_content_count) noexcept;

result<std::vector<std::byte>> build_cuda_rdata_pool_image(
    uint32_t pe_index, size_t pe_count, size_t rdata_pool_size,
    size_t thread_local_rdata_pool_size, size_t block_local_rdata_pool_size,
    std::span<const std::byte> rdata,
    std::span<const cuda_rdata_content_entry> thread_local_rdata_entries,
    std::span<const cuda_rdata_content_entry> block_local_rdata_entries)
    noexcept;

} // namespace detail

END_NS_NNCASE_RT_MODULE
