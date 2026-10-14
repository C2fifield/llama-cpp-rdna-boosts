// wip/moe-upload-overlap: does a concurrent pinned H2D slow a bandwidth-bound compute kernel?
// Proxy for the MoE prefill: the expert GEMMs stream a large weight array from HBM, while the
// staging ring writes ~136 MiB/table into device memory on the copy stream.  If overlap were free,
// (compute + H2D) == compute; the campaign's prefetch result implies it is not.
//
// build:  /opt/rocm-7.14-gfx120X/bin/hipcc -O2 -o /tmp/contendprobe contendprobe.cpp
// run:    LD_LIBRARY_PATH=/opt/rocm-7.14-gfx120X/lib /tmp/contendprobe
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Read a big array many times (HBM read bound), accumulate into out.
__global__ void read_bw(const float * __restrict__ a, size_t n, int iters, float * out) {
    float acc = 0.f;
    for (int it = 0; it < iters; ++it) {
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            acc += a[i];
        }
    }
    if (acc == 12345.678f) {  // never true; keeps the reads alive
        out[threadIdx.x] = acc;
    }
}

int main() {
    hipSetDevice(0);
    const size_t a_bytes = 2ull * 1024 * 1024 * 1024;   // 2 GiB "weights"
    const size_t a_n     = a_bytes / sizeof(float);
    const size_t copy_bytes = 136ull * 1024 * 1024;     // one ring slot / device

    float * a = nullptr;
    hipMalloc(&a, a_bytes);
    hipMemset(a, 1, a_bytes);

    void * dst = nullptr;  hipMalloc(&dst, copy_bytes);
    void * pin = nullptr;  hipHostMalloc(&pin, copy_bytes, hipHostMallocDefault);
    memset(pin, 1, copy_bytes);

    float * out = nullptr; hipMalloc(&out, 4096);
    hipStream_t s_compute, s_copy;
    hipStreamCreate(&s_compute);
    hipStreamCreate(&s_copy);

    const int iters = 24;
    const int blocks = 2048, threads = 256;

    auto run_compute = [&]() {
        read_bw<<<blocks, threads, 0, s_compute>>>(a, a_n, iters, out);
    };
    auto timer = [&](const char * lbl, bool with_copy, bool two_stream) {
        // warmup
        run_compute(); hipStreamSynchronize(s_compute);
        double best = 1e9;
        for (int i = 0; i < 5; ++i) {
            const double t0 = now_ms();
            if (!with_copy) {
                run_compute();
                hipStreamSynchronize(s_compute);
            } else if (!two_stream) {
                run_compute();
                hipMemcpyAsync(dst, pin, copy_bytes, hipMemcpyHostToDevice, s_compute);
                hipStreamSynchronize(s_compute);
            } else {
                run_compute();
                hipMemcpyAsync(dst, pin, copy_bytes, hipMemcpyHostToDevice, s_copy);
                hipStreamSynchronize(s_copy);
                hipStreamSynchronize(s_compute);
            }
            best = std::min(best, now_ms() - t0);
        }
        printf("  %-34s %8.2f ms\n", lbl, best);
    };

    // copy alone
    {
        hipMemcpyAsync(dst, pin, copy_bytes, hipMemcpyHostToDevice, s_copy);
        hipStreamSynchronize(s_copy);
        double best = 1e9;
        for (int i = 0; i < 5; ++i) {
            const double t0 = now_ms();
            hipMemcpyAsync(dst, pin, copy_bytes, hipMemcpyHostToDevice, s_copy);
            hipStreamSynchronize(s_copy);
            best = std::min(best, now_ms() - t0);
        }
        printf("  %-34s %8.2f ms  (%.1f GB/s)\n", "H2D alone (136 MiB, pinned)",
               best, copy_bytes / 1e9 / (best / 1e3));
    }

    timer("compute alone",                       false, false);
    timer("compute + H2D same stream",           true,  false);
    timer("compute + H2D copy stream (overlap)", true,  true);

    // 2-D strided variant, matching the model's gather geometry (256 x 557056, pitch 1114112)
    const size_t w = 557056, pitch = 1114112, rows = 256;
    void * src2 = nullptr; hipHostMalloc(&src2, pitch * rows, hipHostMallocDefault);
    memset(src2, 1, pitch * rows);
    void * dst2 = nullptr; hipMalloc(&dst2, w * rows);
    {
        auto run2 = [&](bool overlap) {
            double best = 1e9;
            for (int i = 0; i < 5; ++i) {
                const double t0 = now_ms();
                run_compute();
                hipMemcpy2DAsync(dst2, w, src2, pitch, w, rows,
                                 hipMemcpyHostToDevice, overlap ? s_copy : s_compute);
                if (overlap) { hipStreamSynchronize(s_copy); }
                hipStreamSynchronize(s_compute);
                best = std::min(best, now_ms() - t0);
            }
            return best;
        };
        hipDeviceSynchronize();
        const double serial = run2(false);
        const double ovl    = run2(true);
        printf("  %-34s %8.2f ms\n", "compute + 2D-H2D same stream", serial);
        printf("  %-34s %8.2f ms\n", "compute + 2D-H2D copy stream", ovl);
    }

    return 0;
}
