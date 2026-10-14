#ifndef GUFO_MODELS_MIMO2_KERNELS_ROCM_COMMUNICATOR_HPP_
#define GUFO_MODELS_MIMO2_KERNELS_ROCM_COMMUNICATOR_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace gufo::models::mimo2::rocm {

/// The exchange ring the transport serves collectives from: `kWindows`
/// windows of `kWindowBytes` each, used in turn (three, so the add reading an
/// overlapped exchange's partial may still be queued when the exchange after
/// the next one lands). A window starts with the writer's header; the partial
/// follows at kDataOffset.
inline constexpr std::size_t kWindows = 3;
inline constexpr std::size_t kWindowBytes = std::size_t{32} << 20;
/// The exchange id's window payload offset: exchange `id` lives at
/// `ExchangeWindowOffset(id)` in the send and receive buffers. Device kernels
/// recover it from a pinned slot holding `id + 1`, so a captured graph picks
/// the same window as the host half replaying that exchange.
[[nodiscard]] inline constexpr std::size_t ExchangeWindowOffset(
    std::uint64_t operation_id) noexcept {
  return static_cast<std::size_t>(operation_id % kWindows) * kWindowBytes;
}

/// Ordered communication boundary used by the model executor. A two-rank sum
/// is one `ExchangePartial` followed by adding the returned peer partial to
/// this rank's partial on the GPU. Each rank adds local plus peer, and float
/// addition of two operands is commutative, so both ranks hold identical sums.
///
/// A distributed communicator requires one exclusive operation scope for the
/// lifetime of a control request. The scope is an opaque logical identity; the
/// current TP2 serving path uses the control command sequence as that value.
/// The communicator adds and validates an operation ordinal for every
/// exchange, so a collective cannot be silently associated with a different
/// request.
class Communicator {
public:
  virtual ~Communicator() = default;
  [[nodiscard]] virtual std::uint32_t rank() const noexcept { return 0; }
  [[nodiscard]] virtual std::uint32_t world_size() const noexcept { return 1; }
  [[nodiscard]] virtual int device_index() const noexcept { return 0; }
  /// The transport's settings, for the log; empty when there is none.
  [[nodiscard]] virtual std::string Describe() const { return {}; }
  [[nodiscard]] virtual bool BeginOperation(std::uint64_t scope_id,
                                            std::string* error) = 0;
  [[nodiscard]] virtual bool EndOperation(std::uint64_t scope_id,
                                          std::string* error) = 0;
  /// Sends this rank's `bytes` of `data`, complete once `stream` drains, and
  /// returns a GPU-readable pointer to the peer's partial of the same size, or
  /// null on failure. Every exchange must use the same stream, and the add
  /// that reads the peer's partial must be queued on it before the second
  /// exchange after this one starts: that exchange reuses the buffer.
  [[nodiscard]] virtual const float* ExchangePartial(const float* data,
                                                     std::size_t bytes,
                                                     hipStream_t stream,
                                                     std::string* error) = 0;
  /// The overlapped form of `ExchangePartial`, so the GPU computes other rows
  /// while a partial is in flight. `StartPartial` queues the exchange of this
  /// rank's partial, complete once `stream` reaches this point, and returns at
  /// once. `FinishPartial` waits for the oldest queued exchange, which must
  /// carry `bytes`, and returns the peer's partial. Both forms share one
  /// sequence and the rule above; finish every queued exchange before the next
  /// `ExchangePartial` or `EndOperation`.
  [[nodiscard]] virtual bool StartPartial(const float* data, std::size_t bytes,
                                          hipStream_t stream,
                                          std::string* error) = 0;
  [[nodiscard]] virtual const float* FinishPartial(std::size_t bytes,
                                                   std::string* error) = 0;

  /// The GPU's half of a queued exchange (QueuePartial). Queue on the stream,
  /// in this order: copy the partial to `send` and then publish `value` in
  /// `ready`, visibly to the host (StagePartial); hold the stream until
  /// `arrived` reaches `value` (WaitValue, giving up after `wait_ticks` of the
  /// device wall clock and setting `timed_out`); then add `peer`. All
  /// pointers are device addresses.
  struct QueuedPartial {
    float* send{nullptr};
    const float* peer{nullptr};
    std::uint64_t* ready{nullptr};
    const std::uint64_t* arrived{nullptr};
    std::uint64_t value{0};
    std::uint32_t* arrivals{nullptr};  ///< StagePartial's block counter
    std::uint32_t* timed_out{nullptr};
    std::uint64_t wait_ticks{0};
  };
  /// Queues an exchange of `bytes` without blocking the caller: a background
  /// thread sends this rank's partial once the stream publishes it and
  /// releases the stream's wait once the peer's partial has arrived, so the
  /// caller keeps queuing work. Shares the sequence and rules above. A failed
  /// exchange releases every wait and is reported by CheckQueued; the stream
  /// then carries meaningless sums to the end of the operation.
  [[nodiscard]] virtual bool QueuePartial(std::size_t bytes,
                                          QueuedPartial* partial,
                                          std::string* error) = 0;
  /// After the stream has drained: false with the first failure of a queued
  /// exchange.
  [[nodiscard]] virtual bool CheckQueued(std::string* error) = 0;

  /// Graph-capture protocol. A stream capture cannot replay the background
  /// thread's exchange, so a capturing executor stages the same kernels but
  /// takes the exchange's host half from a stream host function (replayed by
  /// the graph) and reads the readiness/arrival targets through a pinned
  /// slot the host advances once per replay. `host_func_data` points at
  /// storage the communicator owns for at least as long as the graph.
  struct CapturedPartial {
    QueuedPartial queued;
    std::uint64_t* value_slot{nullptr};  ///< pinned; holds the exchange id
    void* host_func_data{nullptr};
  };
  /// Whether this transport implements the capture protocol below.
  [[nodiscard]] virtual bool CaptureSafe() const { return false; }
  /// Registers one exchange of `bytes` inside the capture session opened by
  /// BeginCaptureSession; fills the pointers the stream kernels and host
  /// function consume.
  [[nodiscard]] virtual bool CapturePartial(std::size_t bytes,
                                            CapturedPartial* out,
                                            std::string* error) {
    (void)bytes;
    (void)out;
    (void)error;
    return false;
  }
  /// The host half of one captured exchange, invoked from the stream host
  /// function: wait for the staged partial, post it, wait for the peer's,
  /// and release the stream's wait. Failures surface through CheckQueued.
  virtual void RunCapturedExchange(const CapturedPartial& partial) {}
  /// Capture-session bracketing; EndCaptureSession returns an opaque handle
  /// plus the number of registered exchanges.
  [[nodiscard]] virtual void* BeginCaptureSession(std::string* error) {
    (void)error;
    return nullptr;
  }
  [[nodiscard]] virtual void* EndCaptureSession(std::uint64_t* count,
                                                std::string* error) {
    (void)error;
    *count = 0;
    return nullptr;
  }
  /// Advances every slot of a finished session by its exchange count, so the
  /// next replay of its graph publishes fresh targets. Call once per replay,
  /// on every rank, before the launch.
  [[nodiscard]] virtual bool AdvanceCapturedSession(void* session,
                                                    std::string* error) {
    (void)session;
    (void)error;
    return false;
  }
};

}  // namespace gufo::models::mimo2::rocm

#endif  // GUFO_MODELS_MIMO2_KERNELS_ROCM_COMMUNICATOR_HPP_
