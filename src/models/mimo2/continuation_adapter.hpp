#ifndef GUFO_MODELS_MIMO2_CONTINUATION_ADAPTER_HPP_
#define GUFO_MODELS_MIMO2_CONTINUATION_ADAPTER_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/cache/adapter.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/mimo2/engine.hpp"

namespace gufo::models::mimo2 {

/// Upstream component cache over the MiMo2 session snapshot layout.
///
/// The engine's snapshot is position-scaled (tokens + per-layer KV rows + a
/// fixed window for ring layers), so it maps onto the component cache as:
/// token ids and the full-attention KV planes as append-row components, and
/// the fixed-shape parts (host header + logits, ring windows, draft state,
/// kept trunk rows) as private components. Two one-byte "frontier" append
/// components carry the kept-row and draft positions that the framework's
/// private positions cannot (private positions attest the checkpoint
/// boundary). All transfers are host-side: one SaveSnapshot per capture
/// epoch is sliced into the framework's buffers, and a restore assembles the
/// exact payload the engine's own RestoreSnapshot consumes.
class ContinuationAdapter final : public cache::Adapter {
public:
  ContinuationAdapter(std::shared_ptr<Model> model, core::SessionMode mode,
                      std::uint32_t context, cache::Identity identity);
  ~ContinuationAdapter() override;

  /// The slot's session, ready for execution.
  [[nodiscard]] Session& GetSession(cache::Slot&);
  /// Same session, shared with the runner state bound to the slot.
  [[nodiscard]] std::shared_ptr<Session> SharedSession(cache::Slot&);

  // cache::Adapter
  [[nodiscard]] cache::Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override;
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard&) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override;
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(const cache::Slot&,
                                                     cache::Rows first,
                                                     cache::Rows end) const
      override;
  void BeginRestore(cache::Slot&,
                    std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot&,
                                                 cache::ComponentId,
                                                 std::span<std::byte>,
                                                 cache::Stream&) override;
  [[nodiscard]] cache::Completion CapturePrivatePiece(
      const cache::Slot&, cache::ComponentId, std::size_t,
      std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                              std::span<const std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivatePiece(
      cache::Slot&, cache::ComponentId, std::size_t,
      std::span<const std::byte>, cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsOut(const cache::Slot&,
                                              cache::ComponentId, cache::Rows,
                                              cache::Rows, std::span<std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows, cache::Rows,
                                             std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

  /// Bytes of the largest single capture payload (backing-class sizing).
  [[nodiscard]] std::size_t LargestPayloadBytes() const noexcept;

private:
  enum class Part : std::uint8_t {
    kTokens,
    kHostRest,
    kPayloadHeader,
    kLayerK,
    kLayerV,
    kMtpK,
    kMtpV,
    kMtpResidual,
    kKeptHidden,
    kHiddenFrontier,
    kMtpFrontier,
  };
  struct Component {
    cache::ComponentDescriptor descriptor;
    Part part{Part::kTokens};
    std::uint32_t index{0};  ///< trunk layer or MTP block number
    bool swa{false};
    std::size_t row_bytes{0};    ///< append rows
    std::size_t state_bytes{0};  ///< private payload
  };
  /// One contiguous piece of a component's payload and where it lands in the
  /// engine payload (executor-region offsets are executor-relative).
  struct Placed {
    std::size_t component{0};
    std::uint64_t component_offset{0};
    std::uint64_t payload_offset{0};
    std::uint64_t bytes{0};
    bool host_source{false};
  };
  struct Shape {
    std::uint32_t position{0};
    std::uint32_t hidden_rows{0};
    std::uint32_t mtp_position{0};
  };
  class State;

  [[nodiscard]] State& As(cache::Slot&) const;
  [[nodiscard]] const State& As(const cache::Slot&) const;
  [[nodiscard]] const Component& Comp(cache::ComponentId id) const;
  [[nodiscard]] Shape ShapeOf(const Session& session) const;
  /// Host-section bytes (session header + tokens + logits) for a position.
  [[nodiscard]] std::size_t HostBytes(std::uint32_t position) const;
  /// One SaveSnapshot per capture epoch, keyed on the session position.
  [[nodiscard]] const SessionSnapshot& CaptureBlob(State& state) const;
  /// Component payload pieces in blob coordinates for a shape.
  [[nodiscard]] std::vector<Placed> PrivatePieces(std::size_t index,
                                                  const Session* session,
                                                  const Shape& shape) const;
  /// Engine payload (executor-section) byte total for a shape.
  [[nodiscard]] std::uint64_t PayloadBytes(const Session* session,
                                           const Shape& shape) const;
  /// Every component slice the engine payload carries for a shape, in walk
  /// order. `session` supplies ring sizes so wrapped windows split exactly as
  /// SaveSnapshot writes them; null yields one run per window.
  [[nodiscard]] std::vector<Placed> Place(const Session* session,
                                          const Shape& shape) const;

  std::shared_ptr<Model> model_;
  core::SessionMode mode_;
  std::uint32_t context_{0};
  cache::Identity identity_;
  std::vector<Component> components_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  std::size_t largest_payload_{0};
  std::uint32_t max_speculative_{0};
  std::uint32_t hidden_size_{0};
  std::uint32_t vocab_{0};
  bool has_mtp_{false};
};

}  // namespace gufo::models::mimo2

#endif  // GUFO_MODELS_MIMO2_CONTINUATION_ADAPTER_HPP_