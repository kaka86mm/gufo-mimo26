#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_RDMA_GID_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_RDMA_GID_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>

namespace gufo::models::qwen38_flash_next::distributed {

enum class RdmaGidType { kInfiniBand, kRoceV1, kRoceV2 };

/// One entry of an RDMA port's GID table.
struct RdmaGidEntry {
  std::uint32_t index{0};
  RdmaGidType type{RdmaGidType::kInfiniBand};
  std::array<std::uint8_t, 16> gid{};
};

/// Chooses the GID a RoCE port uses when none is configured: the RoCE v2 GID
/// of the port netdev's only IPv4 address, or else the port's only RoCE v2
/// GID. Unset (all-zero) entries are skipped. Several candidates are
/// ambiguous and fail with their indices, as does a table without one.
[[nodiscard]] std::optional<std::uint32_t> SelectRoceV2Gid(
    std::span<const RdmaGidEntry> entries, std::string* error_msg);

}  // namespace gufo::models::qwen38_flash_next::distributed

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_RDMA_GID_HPP_
