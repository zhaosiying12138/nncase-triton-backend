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
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>

using namespace nncase;
using namespace nncase::runtime::cuda::detail;

namespace {

void append_u64(std::vector<std::byte> &bytes, uint64_t value) {
    for (size_t i = 0; i < sizeof(value); i++) {
        bytes.push_back(static_cast<std::byte>((value >> (i * 8)) & 0xff));
    }
}

void append_text(std::vector<std::byte> &bytes, std::string_view text) {
    auto ptr = reinterpret_cast<const std::byte *>(text.data());
    bytes.insert(bytes.end(), ptr, ptr + text.size());
}

std::vector<std::byte>
make_multi_content_section(std::span<const std::string_view> contents) {
    std::vector<std::byte> bytes;
    auto header_size = contents.size() * sizeof(uint64_t) * 2;
    std::vector<std::byte> body;

    for (auto content : contents) {
        append_u64(bytes, body.size());
        append_u64(bytes, content.size());
        append_text(body, content);
    }

    if (bytes.size() != header_size) {
        std::abort();
    }

    bytes.insert(bytes.end(), body.begin(), body.end());
    return bytes;
}

bool expect(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << "\n";
        return false;
    }

    return true;
}

bool image_contains(std::span<const std::byte> image, size_t offset,
                    std::string_view text) {
    if (offset + text.size() > image.size()) {
        return false;
    }

    return std::memcmp(image.data() + offset, text.data(), text.size()) == 0;
}

bool test_combined_image_uses_per_pe_and_per_block_contents() {
    std::string_view thread_contents[] = {"t0", "t1", "t2", "t3"};
    std::string_view block_contents[] = {"block0", "block1"};
    auto thread_section = make_multi_content_section(thread_contents);
    auto block_section = make_multi_content_section(block_contents);
    std::byte rdata_bytes[] = {std::byte{'r'}, std::byte{'d'}};

    auto thread_entries_r = parse_cuda_rdata_multi_content_section(
        thread_section, 4, 3);
    if (!expect(thread_entries_r.is_ok(), "thread section did not parse")) {
        return false;
    }

    auto block_count_r =
        infer_cuda_rdata_multi_content_count(block_section, 6, 4);
    if (!expect(block_count_r.is_ok(), "block content count did not infer")) {
        return false;
    }
    if (!expect(block_count_r.unwrap() == 2,
                "block content count should be inferred as 2")) {
        return false;
    }

    auto block_entries_r = parse_cuda_rdata_multi_content_section(
        block_section, block_count_r.unwrap(), 6);
    if (!expect(block_entries_r.is_ok(), "block section did not parse")) {
        return false;
    }

    auto image_r = build_cuda_rdata_pool_image(
        2, 4, 4, 3, 6, rdata_bytes, thread_entries_r.unwrap(),
        block_entries_r.unwrap());
    if (!expect(image_r.is_ok(), "combined image did not build")) {
        return false;
    }

    auto image = image_r.unwrap();
    return expect(image.size() == 13, "combined image size is wrong") &&
           expect(image_contains(image, 0, "rd"), "rdata bytes missing") &&
           expect(image_contains(image, 4, "t2"),
                  "PE 2 thread-local rdata missing") &&
           expect(image_contains(image, 7, "block1"),
                  "PE 2 block-local rdata missing");
}

bool test_block_entries_must_evenly_partition_pes() {
    std::string_view block_contents[] = {"b0", "b1"};
    auto block_section = make_multi_content_section(block_contents);
    auto block_entries_r =
        parse_cuda_rdata_multi_content_section(block_section, 2, 2);
    if (!expect(block_entries_r.is_ok(), "block section did not parse")) {
        return false;
    }

    auto image_r = build_cuda_rdata_pool_image(
        2, 3, 0, 0, 2, {}, {}, block_entries_r.unwrap());
    return expect(image_r.is_err(),
                  "block entries should reject uneven PE partitioning");
}

} // namespace

int main() {
    bool ok = true;
    ok &= test_combined_image_uses_per_pe_and_per_block_contents();
    ok &= test_block_entries_must_evenly_partition_pes();
    return ok ? 0 : 1;
}
