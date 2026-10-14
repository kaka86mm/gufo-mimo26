#include "src/models/qwen38_flash_next/distributed/rdma_gid.hpp"

#include <algorithm>
#include <vector>

namespace gufo::models::qwen38_flash_next::distributed {
namespace {

bool Unset(const std::array<std::uint8_t, 16>& gid) {
  return std::all_of(gid.begin(), gid.end(),
                     [](std::uint8_t byte) { return byte == 0; });
}

/// An IPv4-mapped GID, ::ffff:a.b.c.d, which RoCE v2 derives from an IPv4
/// address of the port's netdev.
bool Ipv4Mapped(const std::array<std::uint8_t, 16>& gid) {
  return std::all_of(gid.begin(), gid.begin() + 10,
                     [](std::uint8_t byte) { return byte == 0; }) &&
         gid[10] == 0xff && gid[11] == 0xff;
}

std::string Indices(const std::vector<std::uint32_t>& indices) {
  std::string text;
  for (const std::uint32_t index : indices) {
    text += (text.empty() ? "" : ", ") + std::to_string(index);
  }
  return text;
}

}  // namespace

std::optional<std::uint32_t> SelectRoceV2Gid(
    std::span<const RdmaGidEntry> entries, std::string* error_msg) {
  std::vector<std::uint32_t> ipv4;
  std::vector<std::uint32_t> any;
  for (const RdmaGidEntry& entry : entries) {
    if (entry.type != RdmaGidType::kRoceV2 || Unset(entry.gid)) {
      continue;
    }
    any.push_back(entry.index);
    if (Ipv4Mapped(entry.gid)) {
      ipv4.push_back(entry.index);
    }
  }
  const std::vector<std::uint32_t>& candidates = ipv4.empty() ? any : ipv4;
  if (candidates.size() == 1) {
    return candidates.front();
  }
  if (error_msg != nullptr) {
    *error_msg =
        candidates.empty()
            ? "the RDMA port has no RoCE v2 GID; give its netdev an address"
            : std::string("the RDMA port has several RoCE v2 ") +
                  (ipv4.empty() ? "GIDs" : "IPv4 GIDs") + " (indices " +
                  Indices(candidates) + "): choose one with --tp-gid-index";
  }
  return std::nullopt;
}

}  // namespace gufo::models::qwen38_flash_next::distributed
