#include "src/models/qwen38_flash_next/distributed/rdma_gid.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using gufo::models::qwen38_flash_next::distributed::RdmaGidEntry;
using gufo::models::qwen38_flash_next::distributed::RdmaGidType;
using gufo::models::qwen38_flash_next::distributed::SelectRoceV2Gid;

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

std::array<std::uint8_t, 16> LinkLocal(std::uint8_t last) {
  std::array<std::uint8_t, 16> gid{};
  gid[0] = 0xfe;
  gid[1] = 0x80;
  gid[15] = last;
  return gid;
}

std::array<std::uint8_t, 16> Ipv4(std::uint8_t last) {
  std::array<std::uint8_t, 16> gid{};
  gid[10] = 0xff;
  gid[11] = 0xff;
  gid[12] = 192;
  gid[13] = 0;
  gid[14] = 2;
  gid[15] = last;
  return gid;
}

}  // namespace

int main() {
  std::string error;

  // A port with a link-local and an IPv4 address, each as RoCE v1 and v2,
  // as irdma and mlx5 list them: the v2 GID of the IPv4 address wins.
  const std::vector<RdmaGidEntry> typical{
      {0, RdmaGidType::kRoceV1, LinkLocal(1)},
      {1, RdmaGidType::kRoceV2, LinkLocal(1)},
      {2, RdmaGidType::kRoceV1, Ipv4(1)},
      {3, RdmaGidType::kRoceV2, Ipv4(1)},
  };
  Require(SelectRoceV2Gid(typical, &error) == 3U,
          "the RoCE v2 GID of the IPv4 address is chosen");

  // Without an IPv4 address the only RoCE v2 GID is used.
  const std::vector<RdmaGidEntry> link_local{
      {0, RdmaGidType::kRoceV1, LinkLocal(1)},
      {1, RdmaGidType::kRoceV2, LinkLocal(1)},
      {2, RdmaGidType::kRoceV2, {}},
  };
  Require(SelectRoceV2Gid(link_local, &error) == 1U,
          "the only RoCE v2 GID is chosen and unset entries are skipped");

  const std::vector<RdmaGidEntry> two_addresses{
      {1, RdmaGidType::kRoceV2, LinkLocal(1)},
      {3, RdmaGidType::kRoceV2, Ipv4(1)},
      {5, RdmaGidType::kRoceV2, Ipv4(2)},
  };
  error.clear();
  Require(!SelectRoceV2Gid(two_addresses, &error).has_value(),
          "two IPv4 addresses are ambiguous");
  Require(error.find("indices 3, 5") != std::string::npos,
          "the ambiguity names the candidates");

  const std::vector<RdmaGidEntry> v1_only{
      {0, RdmaGidType::kRoceV1, LinkLocal(1)},
      {1, RdmaGidType::kInfiniBand, LinkLocal(2)},
  };
  error.clear();
  Require(!SelectRoceV2Gid(v1_only, &error).has_value() && !error.empty(),
          "a table without a RoCE v2 GID fails");
  Require(!SelectRoceV2Gid({}, &error).has_value(), "an empty table fails");

  std::printf("PASS: RoCE v2 GID selection\n");
  return 0;
}
