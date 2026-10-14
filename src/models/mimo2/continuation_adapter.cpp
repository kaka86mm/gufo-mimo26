#include "src/models/mimo2/continuation_adapter.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "src/models/mimo2/kernels/rocm/executor.hpp"

namespace gufo::models::mimo2 {
namespace {

using cache::Rows;

/// Payload rows per append chunk. 1024 rows keep the KV chunks near the
/// backing pool's 64 KiB block multiple for the production geometries and
/// the token chunks small enough to share widely.
constexpr std::size_t kRowsPerChunk = 1024;

/// Host-only transfers settle before returning.
class ReadyCompletion final : public cache::CompletionSignal {
public:
  [[nodiscard]] bool Ready() const noexcept override { return true; }
  [[nodiscard]] cache::TransferResult Wait() noexcept override {
    return cache::TransferResult::kSucceeded;
  }
};

cache::Completion Ready() {
  return cache::Completion(std::make_unique<ReadyCompletion>());
}

/// Merged coverage over blob-relative byte ranges.
void AddCoverage(std::vector<std::pair<std::size_t, std::size_t>>& ranges,
                 std::size_t first, std::size_t end) {
  if (first >= end)
    return;
  ranges.emplace_back(first, end);
  std::ranges::sort(ranges);
}

bool Covered(const std::vector<std::pair<std::size_t, std::size_t>>& ranges,
             std::size_t first, std::size_t end) {
  if (first >= end)
    return true;
  std::size_t frontier = first;
  for (const auto& [begin, last] : ranges) {
    if (begin > frontier)
      break;
    frontier = std::max(frontier, last);
    if (frontier >= end)
      return true;
  }
  return false;
}

}  // namespace

class ContinuationAdapter::State final : public cache::Slot {
public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::shared_ptr<Session> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {}

  [[nodiscard]] bool IsValid() const noexcept override {
    return !restoring && !failed && session != nullptr && session->IsValid();
  }

  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::shared_ptr<Session> session;
  bool restoring{false};
  bool failed{false};
  Shape shape{};
  std::vector<Placed> layout;  ///< restore-time component placements
  std::vector<cache::ComponentPosition> expected;
  std::vector<std::byte> assembly;
  /// Merged covered ranges in blob coordinates (host section + payload).
  std::vector<std::pair<std::size_t, std::size_t>> coverage;
  std::unique_ptr<SessionSnapshot> blob;
  std::uint32_t blob_position{std::numeric_limits<std::uint32_t>::max()};
};

ContinuationAdapter::ContinuationAdapter(std::shared_ptr<Model> model,
                                         core::SessionMode mode,
                                         std::uint32_t context,
                                         cache::Identity identity)
    : model_(std::move(model)),
      mode_(mode),
      context_(context),
      identity_(std::move(identity)) {
  if (model_ == nullptr ||
      (mode_ != core::SessionMode::kAutoregressive &&
       mode_ != core::SessionMode::kSpeculative) ||
      identity_.empty() || context_ == 0 || context_ > model_->MaxContext() ||
      (mode_ == core::SessionMode::kSpeculative && !model_->HasMtp())) {
    throw std::invalid_argument("invalid MiMo component adapter configuration");
  }
  const auto& config = model_->config();
  vocab_ = model_->VocabSize();
  hidden_size_ = config.hidden_size;
  max_speculative_ = model_->executor_->max_speculative();
  has_mtp_ = mode_ == core::SessionMode::kSpeculative;
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(1);  // adapter schema
  put(Session::kSnapshotPayloadVersion);
  put(context_);
  put(has_mtp_ ? 1U : 0U);
  put(max_speculative_);
  put(model_->PrefillCapacity());
  put(model_->DecodeConcurrency());

  const auto push = [&](Part part, cache::ComponentKind kind,
                        std::size_t row_bytes, std::size_t state_bytes,
                        std::uint32_t index) {
    Component c;
    c.part = part;
    c.index = index;
    c.row_bytes = row_bytes;
    c.state_bytes = state_bytes;
    c.descriptor.id =
        cache::ComponentId{static_cast<std::uint32_t>(components_.size() + 1)};
    c.descriptor.layout_version = 1;
    c.descriptor.kind = kind;
    if (kind == cache::ComponentKind::kAppendRows) {
      c.descriptor.row_bytes = row_bytes;
      c.descriptor.rows_per_chunk = kRowsPerChunk;
      largest_payload_ = std::max(largest_payload_, row_bytes * kRowsPerChunk);
    } else {
      c.descriptor.state_bytes = state_bytes;
      largest_payload_ = std::max(largest_payload_, state_bytes);
    }
    components_.push_back(c);
    descriptors_.push_back(c.descriptor);
  };

  push(Part::kTokens, cache::ComponentKind::kAppendRows,
       sizeof(std::int32_t), 0, 0);
  push(Part::kHostRest, cache::ComponentKind::kPrivateState, 0,
       Session::SnapshotHeaderBytes() + std::size_t{vocab_} * sizeof(float), 0);
  push(Part::kPayloadHeader, cache::ComponentKind::kPrivateState, 0,
       rocm::Executor::SnapshotHeaderBytes(), 0);
  const auto layer_components = [&](std::uint32_t layer, Part k_part,
                                    Part v_part, std::uint32_t index) {
    const bool swa = config.LayerIsSwa(layer);
    const auto k_row = std::size_t{config.KvDim(layer)} * sizeof(std::uint16_t);
    const auto v_row = std::size_t{config.VDim(layer)} * sizeof(std::uint16_t);
    if (swa) {
      // Ring layers snapshot only their live window, which is position
      // dependent content of a fixed size: a private component.
      push(k_part, cache::ComponentKind::kPrivateState, 0,
           k_row * config.sliding_window, index);
      push(v_part, cache::ComponentKind::kPrivateState, 0,
           v_row * config.sliding_window, index);
    } else {
      push(k_part, cache::ComponentKind::kAppendRows, k_row, 0, index);
      push(v_part, cache::ComponentKind::kAppendRows, v_row, 0, index);
    }
  };
  for (std::uint32_t il = 0; il < config.num_layers; ++il)
    layer_components(il, Part::kLayerK, Part::kLayerV, il);
  if (has_mtp_) {
    // Only the first block runs the draft chain; the dormant ones hold
    // position zero and contribute no snapshot bytes, so they get no row
    // components at all.
    layer_components(config.num_layers, Part::kMtpK, Part::kMtpV, 0);
    for (std::uint32_t j = 0; j < config.nextn_layers; ++j)
      push(Part::kMtpResidual, cache::ComponentKind::kPrivateState, 0,
           std::size_t{hidden_size_} * sizeof(float), j);
    push(Part::kKeptHidden, cache::ComponentKind::kPrivateState, 0,
         std::size_t{max_speculative_} * hidden_size_ * sizeof(float), 0);
  }
  // One-byte carriers: the framework's private positions attest the
  // checkpoint boundary, so the kept-row and draft positions travel as
  // constant-row append components whose row counts are the payload.
  push(Part::kHiddenFrontier, cache::ComponentKind::kAppendRows, 1, 0, 0);
  if (has_mtp_)
    push(Part::kMtpFrontier, cache::ComponentKind::kAppendRows, 1, 0, 0);
}

ContinuationAdapter::~ContinuationAdapter() = default;

ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("slot belongs to another MiMo adapter");
  return state;
}

const ContinuationAdapter::State& ContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}

const ContinuationAdapter::Component& ContinuationAdapter::Comp(
    cache::ComponentId id) const {
  if (!id.value || id.value > components_.size())
    throw std::invalid_argument("invalid MiMo component ID");
  return components_[id.value - 1];
}

Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("MiMo slot is not ready for execution");
  return *state.session;
}

std::shared_ptr<Session> ContinuationAdapter::SharedSession(cache::Slot& slot) {
  auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("MiMo slot is not ready for execution");
  return state.session;
}

cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = true};
}

std::span<const cache::ComponentDescriptor> ContinuationAdapter::Components()
    const {
  return descriptors_;
}

cache::Identity ContinuationAdapter::CompatibilityIdentity() const {
  return identity_;
}

std::size_t ContinuationAdapter::LargestPayloadBytes() const noexcept {
  return largest_payload_;
}

std::unique_ptr<cache::Slot> ContinuationAdapter::CreateSlot(
    cache::MutationGuard& guard) {
  std::string error;
  auto session = model_->CreateSession(mode_, context_, &error);
  if (session == nullptr)
    throw std::runtime_error("MiMo component slot session: " + error);
  return std::make_unique<State>(*this, guard,
                                 std::shared_ptr<Session>(std::move(session)));
}

ContinuationAdapter::Shape ContinuationAdapter::ShapeOf(
    const Session& session) const {
  Shape shape;
  shape.position = session.Position();
  shape.hidden_rows = session.KeptHiddenRows();
  shape.mtp_position = has_mtp_ ? session.MtpPosition() : 0;
  return shape;
}

std::size_t ContinuationAdapter::HostBytes(std::uint32_t position) const {
  return Session::SnapshotHeaderBytes() +
         std::size_t{position} * sizeof(std::int32_t) +
         std::size_t{vocab_} * sizeof(float);
}

std::vector<ContinuationAdapter::Placed> ContinuationAdapter::Place(
    const Session* session, const Shape& shape) const {
  // The walk is driven by the shape; the session only supplied it.
  (void)session;
  struct WalkEntry {
    std::uint64_t offset{0};
    std::uint64_t bytes{0};
    std::string_view name;
  };
  std::vector<WalkEntry> walk;
  if (!model_->executor_->VisitSnapshotLayout(
          shape.position, shape.hidden_rows, shape.mtp_position, has_mtp_,
          [&](void*, std::uint64_t offset, std::uint64_t bytes,
              const char* what) {
            walk.push_back({offset, bytes, what});
            return true;
          })) {
    throw std::logic_error("MiMo snapshot layout walk failed");
  }
  const auto host = HostBytes(shape.position);
  std::vector<Placed> placed;
  std::size_t next = 0;
  const auto emit = [&](std::size_t component_index, std::uint64_t expected,
                        const char* name) {
    if (expected == 0)
      return;
    std::uint64_t done = 0;
    while (done < expected) {
      if (next >= walk.size())
        throw std::logic_error("MiMo snapshot walk is shorter than planned");
      const auto& entry = walk[next++];
      if (entry.name != name)
        throw std::logic_error("MiMo snapshot region name disagrees");
      placed.push_back(
          Placed{component_index, done, host + entry.offset, entry.bytes, false});
      done += entry.bytes;
    }
    if (done != expected)
      throw std::logic_error("MiMo snapshot region sizes disagree");
  };
  const auto& config = model_->config();
  for (std::size_t i = 0; i < components_.size(); ++i) {
    const auto& c = components_[i];
    switch (c.part) {
      case Part::kPayloadHeader:
        emit(i, c.descriptor.state_bytes, "payload header");
        break;
      case Part::kLayerK:
      case Part::kLayerV: {
        const std::uint64_t rows =
            c.swa ? std::min<std::uint64_t>(shape.position,
                                            config.sliding_window)
                  : shape.position;
        emit(i, rows * c.row_bytes, c.swa ? "cache window" : "cache");
        break;
      }
      case Part::kMtpK:
      case Part::kMtpV: {
        const std::uint64_t rows =
            c.swa ? std::min<std::uint64_t>(shape.mtp_position,
                                            config.sliding_window)
                  : shape.mtp_position;
        emit(i, rows * c.row_bytes, c.swa ? "cache window" : "cache");
        break;
      }
      case Part::kMtpResidual:
        emit(i, std::size_t{hidden_size_} * sizeof(float), "draft residual");
        break;
      case Part::kKeptHidden:
        emit(i, std::uint64_t{shape.hidden_rows} * hidden_size_ * sizeof(float),
             "kept trunk rows");
        break;
      default:
        break;  // tokens, host rest and the frontier carriers are host-side
    }
  }
  if (next != walk.size())
    throw std::logic_error("MiMo snapshot walk carries extra regions");
  return placed;
}

std::uint64_t ContinuationAdapter::PayloadBytes(const Session* session,
                                                const Shape& shape) const {
  const auto placed = Place(session, shape);
  if (placed.empty())
    return 0;
  const auto& last = placed.back();
  return last.payload_offset + last.bytes - HostBytes(shape.position);
}

const SessionSnapshot& ContinuationAdapter::CaptureBlob(State& state) const {
  const auto position = state.session->Position();
  if (!state.blob || state.blob_position != position) {
    std::string error;
    state.blob = state.session->SaveSnapshot(&error);
    if (!state.blob)
      throw std::runtime_error("MiMo component capture failed: " + error);
    state.blob_position = position;
  }
  return *state.blob;
}

/// Component payload pieces in blob coordinates for the current shape.
std::vector<ContinuationAdapter::Placed> ContinuationAdapter::PrivatePieces(
    std::size_t index, const Session* session, const Shape& shape) const {
  const auto& c = components_[index];
  if (c.part == Part::kHostRest) {
    const auto host = HostBytes(shape.position);
    return {
        Placed{index, 0, 0, Session::SnapshotHeaderBytes(), true},
        Placed{index, Session::SnapshotHeaderBytes(),
               Session::SnapshotHeaderBytes() +
                   std::size_t{shape.position} * sizeof(std::int32_t),
               std::size_t{vocab_} * sizeof(float), true},
    };
  }
  std::vector<Placed> pieces;
  for (const auto& placed : Place(session, shape)) {
    if (placed.component == index)
      pieces.push_back(placed);
  }
  return pieces;
}

std::vector<cache::Rows> ContinuationAdapter::PlanPrefill(
    const cache::Slot& slot, cache::Rows first, cache::Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid MiMo prefill plan");
  std::vector<cache::Rows> result;
  const auto capacity = model_->PrefillCapacity();
  while (first < end) {
    const auto remaining = end - first;
    first += std::min<Rows>(remaining, capacity);
    result.push_back(first);
  }
  return result;
}

std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid MiMo slot");
  const auto shape = ShapeOf(*state.session);
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(components_.size());
  for (const auto& c : components_) {
    cache::ComponentPosition position;
    position.id = c.descriptor.id;
    switch (c.part) {
      case Part::kHiddenFrontier:
        position.valid_rows = shape.hidden_rows;
        break;
      case Part::kMtpFrontier:
        position.valid_rows = shape.mtp_position;
        break;
      case Part::kMtpK:
      case Part::kMtpV:
        // Append rows count their own position; private components attest
        // the checkpoint boundary.
        position.valid_rows =
            c.descriptor.kind == cache::ComponentKind::kAppendRows
                ? shape.mtp_position
                : shape.position;
        break;
      default:
        position.valid_rows = shape.position;
        break;
    }
    positions.push_back(position);
  }
  return positions;
}

void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.IsValid() || positions.size() != components_.size())
    throw std::invalid_argument("incomplete MiMo restore positions");
  for (const auto& c : components_) {
    if (std::ranges::count(positions, c.descriptor.id,
                           &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("missing or duplicate MiMo frontier");
  }
  const auto value = [&](Part part) -> std::uint64_t {
    for (std::size_t i = 0; i < components_.size(); ++i)
      if (components_[i].part == part)
        return std::ranges::find(positions, components_[i].descriptor.id,
                                 &cache::ComponentPosition::id)
            ->valid_rows;
    return 0;
  };
  Shape shape;
  shape.position = static_cast<std::uint32_t>(value(Part::kTokens));
  shape.hidden_rows = static_cast<std::uint32_t>(value(Part::kHiddenFrontier));
  shape.mtp_position =
      has_mtp_ ? static_cast<std::uint32_t>(value(Part::kMtpFrontier)) : 0;
  if (shape.position == 0 || shape.position > context_ ||
      shape.hidden_rows > max_speculative_ ||
      shape.hidden_rows > shape.position || shape.mtp_position > shape.position)
    throw std::invalid_argument("MiMo restore frontier is out of range");
  // No borrowers exist (every capture copies), so no guard ranges and no
  // preservation are needed; the whole state is replaced at Validate.
  state.layout = Place(state.session.get(), shape);
  std::uint64_t payload = 0;
  for (const auto& placed : state.layout)
    payload = std::max(payload, placed.payload_offset + placed.bytes -
                                    HostBytes(shape.position));
  state.shape = shape;
  state.expected.assign(positions.begin(), positions.end());
  state.assembly.assign(HostBytes(shape.position) + payload, std::byte{});
  state.coverage.clear();
  state.blob.reset();
  state.blob_position = std::numeric_limits<std::uint32_t>::max();
  state.restoring = true;
}

cache::Completion ContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}

cache::Completion ContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  (void)stream;  // host-side copies settle synchronously
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid())
    throw std::logic_error("cannot capture an invalid MiMo slot");
  const auto& component = Comp(id);
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > component.descriptor.state_bytes ||
      bytes.size() > component.descriptor.state_bytes - offset)
    throw std::invalid_argument("invalid MiMo private capture");
  const auto& blob = CaptureBlob(state);
  const auto source = blob.bytes();
  const auto shape = ShapeOf(*state.session);
  const auto pieces = PrivatePieces(id.value - 1, state.session.get(), shape);
  std::fill(bytes.begin(), bytes.end(), std::byte{});
  for (const auto& piece : pieces) {
    const auto first = std::max(offset, piece.component_offset);
    const auto end = std::min(offset + bytes.size(),
                              piece.component_offset + piece.bytes);
    if (first >= end)
      continue;
    std::memcpy(bytes.data() + (first - offset),
                source.data() + piece.payload_offset +
                    (first - piece.component_offset),
                static_cast<std::size_t>(end - first));
  }
  return Ready();
}

cache::Completion ContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  return LoadPrivatePiece(slot, id, 0, bytes, stream);
}

cache::Completion ContinuationAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  (void)stream;
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("MiMo load requires an unfailed restore");
  try {
    const auto& component = Comp(id);
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > component.descriptor.state_bytes ||
        bytes.size() > component.descriptor.state_bytes - offset)
      throw std::invalid_argument("invalid MiMo private load");
    for (const auto& piece : state.layout) {
      if (piece.component != id.value - 1)
        continue;
      const auto first = std::max(offset, piece.component_offset);
      const auto end = std::min(offset + bytes.size(),
                                piece.component_offset + piece.bytes);
      if (first >= end)
        continue;
      std::memcpy(state.assembly.data() + piece.payload_offset +
                      (first - piece.component_offset),
                  bytes.data() + (first - offset),
                  static_cast<std::size_t>(end - first));
      AddCoverage(state.coverage,
                  static_cast<std::size_t>(piece.payload_offset +
                                           (first - piece.component_offset)),
                  static_cast<std::size_t>(piece.payload_offset +
                                           (end - piece.component_offset)));
    }
    return Ready();
  } catch (...) {
    state.failed = true;
    throw;
  }
}

cache::Completion ContinuationAdapter::CopyRowsOut(
    const cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<std::byte> bytes, cache::Stream& stream) {
  (void)stream;
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid MiMo slot");
  const auto& component = Comp(id);
  const auto row = component.descriptor.row_bytes;
  if (component.descriptor.kind != cache::ComponentKind::kAppendRows || !row ||
      first > end || bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid MiMo row capture");
  if (component.part == Part::kTokens) {
    const auto tokens = state.session->Tokens();
    if (end > tokens.size())
      throw std::invalid_argument("MiMo token capture exceeds the position");
    std::memcpy(bytes.data(), tokens.data() + first,
                static_cast<std::size_t>(end - first) * sizeof(std::int32_t));
    return Ready();
  }
  if (component.part == Part::kHiddenFrontier ||
      component.part == Part::kMtpFrontier) {
    // Constant rows; only the row count is meaningful.
    std::fill(bytes.begin(), bytes.end(), std::byte{});
    return Ready();
  }
  // Full-attention KV planes slice straight out of the capture blob.
  const auto shape = ShapeOf(*state.session);
  const auto& blob = CaptureBlob(state);
  const auto source = blob.bytes();
  std::uint64_t region = 0;
  for (const auto& placed : Place(state.session.get(), shape)) {
    if (placed.component == id.value - 1) {
      region = placed.payload_offset;
      break;
    }
  }
  if (region == 0)
    throw std::logic_error("MiMo row component has no payload region");
  std::memcpy(bytes.data(), source.data() + region + first * row,
              static_cast<std::size_t>(end - first) * row);
  return Ready();
}

cache::Completion ContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  (void)stream;
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("MiMo load requires an unfailed restore");
  try {
    const auto& component = Comp(id);
    const auto row = component.descriptor.row_bytes;
    if (component.descriptor.kind != cache::ComponentKind::kAppendRows || !row ||
        first > end || bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid MiMo row load");
    if (component.part == Part::kHiddenFrontier ||
        component.part == Part::kMtpFrontier)
      return Ready();  // constant rows have no payload representation
    std::uint64_t target = 0;
    if (component.part == Part::kTokens) {
      target = Session::SnapshotHeaderBytes() + first * sizeof(std::int32_t);
    } else {
      for (const auto& placed : state.layout) {
        if (placed.component == id.value - 1) {
          target = placed.payload_offset;
          break;
        }
      }
      if (target == 0)
        throw std::logic_error("MiMo row component has no payload region");
      target += first * row;
    }
    if (target + bytes.size() > state.assembly.size())
      throw std::invalid_argument("MiMo row load exceeds the payload");
    std::memcpy(state.assembly.data() + target, bytes.data(), bytes.size());
    AddCoverage(state.coverage, static_cast<std::size_t>(target),
                static_cast<std::size_t>(target + bytes.size()));
    return Ready();
  } catch (...) {
    state.failed = true;
    throw;
  }
}

bool ContinuationAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("MiMo validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || positions.size() != state.expected.size())
    return refuse();
  for (const auto& expected : state.expected) {
    if (std::ranges::count(positions, expected) != 1)
      return refuse();
  }
  std::ranges::sort(state.coverage);
  const auto host = HostBytes(state.shape.position);
  const auto required = [&](std::size_t first, std::size_t last) {
    return Covered(state.coverage, first, last);
  };
  if (!required(0, Session::SnapshotHeaderBytes()) ||
      !required(Session::SnapshotHeaderBytes(), host) ||
      !required(host, state.assembly.size()))
    return refuse();
  std::string error;
  if (!state.session->RestoreSnapshot(
          std::span<const std::uint8_t>(
              reinterpret_cast<const std::uint8_t*>(state.assembly.data()),
              state.assembly.size()),
          &error)) {
    state.failed = true;
    return false;
  }
  state.restoring = false;
  state.expected.clear();
  state.layout.clear();
  state.assembly.clear();
  state.assembly.shrink_to_fit();
  state.coverage.clear();
  state.blob.reset();
  state.blob_position = std::numeric_limits<std::uint32_t>::max();
  return true;
}

bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  try {
    auto& state = As(slot);
    state.session->Reset();
    state.failed = false;
    state.restoring = false;
    state.expected.clear();
    state.layout.clear();
    state.assembly.clear();
    state.coverage.clear();
    state.blob.reset();
    state.blob_position = std::numeric_limits<std::uint32_t>::max();
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::mimo2