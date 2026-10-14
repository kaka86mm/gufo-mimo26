// SPDX-License-Identifier: MIT
// Model-private HIP runtime helpers for the imported ggml matrix templates.

#include "common.hip.hpp"   // pulls in mimo2_ggml_stubs.h via redirect headers

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Device properties are queried once and shared by the matrix dispatchers.
const mimo2_ggml_hip_device_info & mimo2_ggml_hip_info() {
    static mimo2_ggml_hip_device_info info;
    static std::once_flag once;
    std::call_once(once, []{
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        if (err != hipSuccess) {
            fprintf(stderr, "mimo2_ggml_hip_info: hipGetDeviceCount failed: %s\n", hipGetErrorString(err));
            count = 0;
        }
        if (count > MIMO2_GGML_HIP_MAX_DEVICES) count = MIMO2_GGML_HIP_MAX_DEVICES;
        info.device_count = count;
        for (int i = 0; i < count; i++) {
            hipDeviceProp_t p;
            MIMO2_HIP_CHECK(hipGetDeviceProperties(&p, i));
            unsigned arch = 0;
            if (sscanf(p.gcnArchName, "gfx%x", &arch) != 1) {
                fprintf(stderr, "mimo2_ggml_hip_info: unsupported HIP architecture '%s'\n", p.gcnArchName);
            }
            info.devices[i].cc                          = MIMO2_GGML_HIP_CC_OFFSET_AMD + (int)arch;
            info.devices[i].nsm                         = p.multiProcessorCount;
            info.devices[i].smpb                        = p.sharedMemPerBlock;
            info.devices[i].smpbo                       = p.sharedMemPerBlockOptin;
            info.devices[i].integrated                  = p.integrated != 0;
            info.devices[i].vmm                         = false;
            info.devices[i].vmm_granularity             = 0;
            info.devices[i].total_vram                  = p.totalGlobalMem;
            info.devices[i].warp_size                   = p.warpSize;
            info.devices[i].supports_cooperative_launch = p.cooperativeLaunch != 0;
        }
    });
    return info;
}

// mimo2_ggml_hip_get_device / mimo2_ggml_hip_set_device are declared (not defined) in
// common.hip.hpp. We provide thin wrappers.

int mimo2_ggml_hip_get_device() {
    int dev = 0;
    hipGetDevice(&dev);
    return dev;
}

void mimo2_ggml_hip_set_device(int device) {
    int cur = -1;
    hipGetDevice(&cur);
    if (cur != device) {
        MIMO2_HIP_CHECK(hipSetDevice(device));
    }
}

int64_t ggml_time_us() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
}

// ----------------------------------------------------------------------------
// mimo2_ggml_hip_error: invoked by the MIMO2_HIP_CHECK / MIMO2_HIPBLAS_CHECK macros defined
// in common.hip.hpp on the error path. Marked [[noreturn]] in the declaration
// (common.hip.hpp:155) - abort() satisfies that contract.
// ----------------------------------------------------------------------------

[[noreturn]] void mimo2_ggml_hip_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "HIP error: %s\n  call: %s\n  in: %s at %s:%d\n", msg, stmt, func, file, line);
    fflush(stderr);
    abort();
}

// ----------------------------------------------------------------------------
// Concrete pool wrapping hipMallocAsync / hipFreeAsync.
// ----------------------------------------------------------------------------

namespace {

/* Thread-local stream that mimo2_naive_pool uses for hipMallocAsync /
 * hipFreeAsync.  Defaults to hipStreamPerThread (preserves prior
 * behaviour).  Step 8 / CUDA Graphs sets this to the capture stream
 * before allocating, so the alloc node lives on the captured stream
 * and capture is not invalidated.  Set via mimo2_pool_set_stream() from
 * mimo2_mmq.cu wrappers; an explicit stream=0 means the legacy default
 * stream so pool ops stay ordered with legacy-stream kernels. */
static thread_local hipStream_t t_mimo2_pool_stream = hipStreamPerThread;

} // anonymous namespace

extern "C" void mimo2_pool_set_stream(hipStream_t stream) {
    t_mimo2_pool_stream = stream;
}

extern "C" hipStream_t mimo2_pool_get_stream(void) {
    return t_mimo2_pool_stream;
}

namespace {

/// Caching pool: a freed block goes to an exact-size free list instead of
/// hipFreeAsync, and a later alloc of the same size reuses it on the same
/// stream. The inference loop repeats a handful of staging shapes per layer,
/// so after the first step nearly every alloc/free pair stays inside this
/// structure — no driver call, no stream-ordered alloc/free churn in the
/// hot path (which also bisects: churn off is the stable configuration).
/// A block freed on a different stream than the current one is released for
/// real instead of cached (reuse across streams would not be ordered).
struct mimo2_naive_pool : public mimo2_ggml_hip_pool {
    struct Cached {
        void * ptr;
        hipStream_t stream;
    };
    int device;
    std::mutex mutex;
    std::unordered_map<size_t, std::vector<Cached>> free_list;
    size_t cached_bytes = 0;
    // Staging is small (single-digit MB per tier); the cap bounds what the
    // cache can hold while comfortably fitting every repeated shape.
    static constexpr size_t kMaxCachedBytes = 512ull << 20;

    explicit mimo2_naive_pool(int device) : device(device) {}

    void * alloc(size_t size, size_t * actual_size) override {
        if (size == 0) {
            if (actual_size) *actual_size = 0;
            return nullptr;
        }
        {
            std::lock_guard lock(mutex);
            const auto it = free_list.find(size);
            if (it != free_list.end()) {
                auto & blocks = it->second;
                for (auto b = blocks.rbegin(); b != blocks.rend(); ++b) {
                    if (b->stream == t_mimo2_pool_stream) {
                        void * ptr = b->ptr;
                        blocks.erase(std::next(b).base());
                        cached_bytes -= size;
                        if (actual_size) *actual_size = size;
                        return ptr;
                    }
                }
            }
        }
        mimo2_ggml_hip_set_device(device);
        void * ptr = nullptr;
        MIMO2_HIP_CHECK(hipMallocAsync(&ptr, size, t_mimo2_pool_stream));
        if (actual_size) *actual_size = size;
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        if (!ptr) return;
        // GUFO_POOL_NOFREE parks every staging buffer (leaks until process
        // exit): bisects whether the stream-ordered alloc/free churn is
        // what HIP_LAUNCH_BLOCKING was hiding on gfx1151.
        static const bool nofree = getenv("GUFO_POOL_NOFREE") != nullptr;
        if (nofree) return;
        // GUFO_POOL_NOCACHE restores the plain hipFreeAsync path.
        static const bool nocache = getenv("GUFO_POOL_NOCACHE") != nullptr;
        if (!nocache && size != 0) {
            std::lock_guard lock(mutex);
            if (cached_bytes + size <= kMaxCachedBytes) {
                free_list[size].push_back(Cached{ptr, t_mimo2_pool_stream});
                cached_bytes += size;
                return;
            }
        }
        mimo2_ggml_hip_set_device(device);
        MIMO2_HIP_CHECK(hipFreeAsync(ptr, t_mimo2_pool_stream));
    }

    ~mimo2_naive_pool() override {
        for (auto & [size, blocks] : free_list) {
            for (const Cached & b : blocks) {
                MIMO2_HIP_CHECK(hipFreeAsync(b.ptr, b.stream));
            }
        }
        free_list.clear();
        cached_bytes = 0;
    }
};

} // anonymous namespace

std::unique_ptr<mimo2_ggml_hip_pool> mimo2_ggml_hip_context::new_pool_for_device(int device, int /*stream_no*/) {
    return std::unique_ptr<mimo2_ggml_hip_pool>(new mimo2_naive_pool(device));
}

mimo2_ggml_hip_context::~mimo2_ggml_hip_context() {
    if (copy_event) {
        hipEventDestroy(copy_event);
        copy_event = nullptr;
    }
    // streams[][], hipblas_handles[], and pools[][] are owned-by-value
    // (hipStream_t and hipblasHandle_t are opaque handles - destroying the
    // context "should" tear them down, but in our shim ds4 manages streams
    // externally and we leave them alone. The pools auto-destruct via
    // unique_ptr.).
}
