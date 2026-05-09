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
#include "runtime_rdata_layout.h"
#include <algorithm>
#include <cstring>
#include <limits>

using namespace nncase;

BEGIN_NS_NNCASE_RT_MODULE(cuda)
namespace detail {

namespace {

constexpr size_t multi_content_entry_size = sizeof(uint64_t) * 2;

bool checked_mul(size_t lhs, size_t rhs, size_t &value) noexcept {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }

    value = lhs * rhs;
    return true;
}

bool checked_add(size_t lhs, size_t rhs, size_t &value) noexcept {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }

    value = lhs + rhs;
    return true;
}

uint64_t read_u64(std::span<const std::byte> bytes, size_t offset) noexcept {
    uint64_t value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

result<std::vector<cuda_rdata_content_entry>>
parse_cuda_rdata_multi_content_section(
    std::span<const std::byte> section, size_t content_count,
    size_t max_content_size, bool require_full_content) noexcept {
    size_t header_size;
    if (!checked_mul(content_count, multi_content_entry_size, header_size) ||
        header_size > section.size_bytes()) {
        return err(std::errc::invalid_argument);
    }

    auto content_size = section.size_bytes() - header_size;
    std::vector<cuda_rdata_content_entry> entries;
    try {
        entries.reserve(content_count);
    } catch (...) {
        return err(std::errc::not_enough_memory);
    }

    size_t previous_end = 0;
    for (size_t i = 0; i < content_count; i++) {
        auto offset64 =
            read_u64(section, i * multi_content_entry_size);
        auto length64 =
            read_u64(section, i * multi_content_entry_size + sizeof(uint64_t));
        if (offset64 > std::numeric_limits<size_t>::max() ||
            length64 > std::numeric_limits<size_t>::max()) {
            return err(std::errc::invalid_argument);
        }

        auto offset = static_cast<size_t>(offset64);
        auto length = static_cast<size_t>(length64);
        if (length > max_content_size || offset > content_size ||
            length > content_size - offset || offset < previous_end) {
            return err(std::errc::invalid_argument);
        }

        previous_end = offset + length;
        entries.push_back(cuda_rdata_content_entry{
            offset, section.subspan(header_size + offset, length)});
    }

    if (require_full_content && previous_end != content_size) {
        return err(std::errc::invalid_argument);
    }

    return ok(std::move(entries));
}

} // namespace

result<std::vector<cuda_rdata_content_entry>>
parse_cuda_rdata_multi_content_section(
    std::span<const std::byte> section, size_t content_count,
    size_t max_content_size) noexcept {
    return parse_cuda_rdata_multi_content_section(
        section, content_count, max_content_size, true);
}

result<size_t> infer_cuda_rdata_multi_content_count(
    std::span<const std::byte> section, size_t max_content_size,
    size_t max_content_count) noexcept {
    if (section.empty()) {
        return ok<size_t>(0);
    }

    auto max_header_count =
        std::min(max_content_count, section.size_bytes() /
                                        multi_content_entry_size);
    for (size_t content_count = max_header_count; content_count != 0;
         content_count--) {
        auto entries_r = parse_cuda_rdata_multi_content_section(
            section, content_count, max_content_size, true);
        if (entries_r.is_ok()) {
            return ok(content_count);
        }
    }

    return err(std::errc::invalid_argument);
}

result<std::vector<std::byte>> build_cuda_rdata_pool_image(
    uint32_t pe_index, size_t pe_count, size_t rdata_pool_size,
    size_t thread_local_rdata_pool_size, size_t block_local_rdata_pool_size,
    std::span<const std::byte> rdata,
    std::span<const cuda_rdata_content_entry> thread_local_rdata_entries,
    std::span<const cuda_rdata_content_entry> block_local_rdata_entries)
    noexcept {
    if (pe_count == 0 || pe_index >= pe_count) {
        return err(std::errc::result_out_of_range);
    }

    // The CUDA bridge exposes one rdata pointer to Python/Triton. Keep the
    // three logical pools in that device allocation in metadata order:
    // .rdata, .thread_local_rdata for this PE, then .block_local_rdata for
    // this PE's block.
    auto rdata_capacity = std::max(rdata_pool_size, rdata.size_bytes());
    size_t thread_local_offset;
    size_t block_local_offset;
    size_t total_size;
    if (!checked_add(rdata_capacity, thread_local_rdata_pool_size,
                     block_local_offset) ||
        !checked_add(block_local_offset, block_local_rdata_pool_size,
                     total_size)) {
        return err(std::errc::value_too_large);
    }
    thread_local_offset = rdata_capacity;

    if (thread_local_rdata_pool_size != 0 &&
        thread_local_rdata_entries.size() != pe_count) {
        return err(std::errc::invalid_argument);
    }

    if (block_local_rdata_pool_size != 0) {
        if (block_local_rdata_entries.empty() ||
            pe_count % block_local_rdata_entries.size() != 0) {
            return err(std::errc::invalid_argument);
        }
    }

    std::vector<std::byte> image;
    try {
        image.resize(total_size);
    } catch (...) {
        return err(std::errc::not_enough_memory);
    }

    if (!rdata.empty()) {
        std::memcpy(image.data(), rdata.data(), rdata.size_bytes());
    }

    if (thread_local_rdata_pool_size != 0) {
        auto entry = thread_local_rdata_entries[pe_index];
        if (entry.bytes.size_bytes() > thread_local_rdata_pool_size) {
            return err(std::errc::invalid_argument);
        }

        if (!entry.bytes.empty()) {
            std::memcpy(image.data() + thread_local_offset,
                        entry.bytes.data(), entry.bytes.size_bytes());
        }
    }

    if (block_local_rdata_pool_size != 0) {
        auto pes_per_block = pe_count / block_local_rdata_entries.size();
        auto block_index = pe_index / pes_per_block;
        auto entry = block_local_rdata_entries[block_index];
        if (entry.bytes.size_bytes() > block_local_rdata_pool_size) {
            return err(std::errc::invalid_argument);
        }

        if (!entry.bytes.empty()) {
            std::memcpy(image.data() + block_local_offset,
                        entry.bytes.data(), entry.bytes.size_bytes());
        }
    }

    return ok(std::move(image));
}

} // namespace detail
END_NS_NNCASE_RT_MODULE
