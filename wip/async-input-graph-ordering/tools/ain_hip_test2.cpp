// Part 2 of the wip/async-input-graph-ordering investigation: does an event recorded on the compute
// stream right after `hipGraphLaunch` actually cover the replayed graph's *execution*?
//
// The delivery's guard is: after graph i completes it records `done_ev` on the compute stream; the next
// upload (copy i+1) waits on `done_ev` before overwriting the input buffer.  If `done_ev` is satisfied
// before the graph has finished, copy i+1 lands while graph i is still reading -> nondeterministic input.
//
// This needs a graph that is SLOWER than the host loop, otherwise the host never runs ahead and the
// defect cannot show.  Each iteration: copy a fresh slot into `dst`, wait on the copy event, launch the
// graph (spin ~SPIN us, then out[ctr++] = dst[0]), record done_ev; the *next* iteration's copy waits on
// done_ev (unless disabled).
//
// modes:
//   0  guard present (copy waits done_ev), slow graph        -> must be all-correct if the guard works
//   1  guard absent                                          -> expected to fail (detector sanity)
//   2  guard present, but the completion event is recorded INSIDE the graph (extra node) *and* outside
//   3  guard present, slow graph, host sleeps 0 (control)

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static constexpr int B = 32;

__global__ void k_spin(int64_t iters, int * sink) {
    int64_t acc = threadIdx.x;
    for (int64_t i = 0; i < iters; ++i) {
        acc = acc * 6364136223846793005ll + 1442695040888963407ll;
    }
    if (acc == 0x5a5a5a5a && blockIdx.x == 0xffffffffu) { *sink = 1; }
}

__global__ void k_read(const int * in, int * out, int * ctr) {
    const int i = atomicAdd(ctr, 1);
    out[i] = in[0];
}

int main(int argc, char ** argv) {
    const int    mode  = argc > 1 ? atoi(argv[1]) : 0;
    const int    N     = argc > 2 ? atoi(argv[2]) : 300;
    const int64_t spin  = argc > 3 ? atoll(argv[3]) : 200000;   // tune so one graph is ~O(100 us)

    hipStream_t compute;
    HC(hipStreamCreateWithFlags(&compute, hipStreamNonBlocking));

    int * dst = nullptr, * out = nullptr, * ctr = nullptr, * sink = nullptr;
    HC(hipMalloc(&dst, B * sizeof(int)));
    HC(hipMalloc(&out, N * sizeof(int)));
    HC(hipMalloc(&ctr, sizeof(int)));
    HC(hipMalloc(&sink, sizeof(int)));
    HC(hipMemset(dst, 0, B * sizeof(int)));
    HC(hipMemset(out, -1, N * sizeof(int)));
    HC(hipMemset(ctr, 0, sizeof(int)));

    // ring: one fresh pinned slot per iteration (never rewritten)
    std::vector<int *> slots(N);
    for (int i = 0; i < N; ++i) {
        HC(hipHostMalloc((void **) &slots[i], B * sizeof(int)));
        for (int j = 0; j < B; ++j) slots[i][j] = i;
    }

    hipEvent_t ev, done_ev;
    HC(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
    HC(hipEventCreateWithFlags(&done_ev, hipEventDisableTiming));

    hipGraph_t graph = nullptr; hipGraphExec_t exec = nullptr;
    HC(hipStreamBeginCapture(compute, hipStreamCaptureModeRelaxed));
    k_spin<<<1, 32, 0, compute>>>(spin, sink);
    k_read<<<1, 1, 0, compute>>>(dst, out, ctr);
    HC(hipStreamEndCapture(compute, &graph));
    HC(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        if (mode == 0 && i > 0) { HC(hipStreamWaitEvent(hipStreamPerThread, done_ev, 0)); }
        HC(hipMemcpyAsync(dst, slots[i], B * sizeof(int), hipMemcpyHostToDevice, hipStreamPerThread));
        HC(hipEventRecord(ev, hipStreamPerThread));
        HC(hipStreamWaitEvent(compute, ev, 0));
        HC(hipGraphLaunch(exec, compute));
        if (mode == 0) { HC(hipEventRecord(done_ev, compute)); }
    }
    const auto t1 = std::chrono::steady_clock::now();
    HC(hipStreamSynchronize(compute));
    HC(hipStreamSynchronize(hipStreamPerThread));
    const auto t2 = std::chrono::steady_clock::now();

    std::vector<int> h(N);
    HC(hipMemcpy(h.data(), out, N * sizeof(int), hipMemcpyDeviceToHost));

    int bad = 0, forward = 0, first_bad = -1, worst = 0;
    for (int i = 0; i < N; ++i) {
        if (h[i] != i) {
            ++bad;
            if (first_bad < 0) first_bad = i;
            const int gap = h[i] - i;
            if (gap > 0) ++forward;
            if (gap > worst) worst = gap;
        }
    }
    const char * names[] = { "guard", "no-guard", "guard+inside", "guard+sleep0" };
    printf("mode=%d (%s) N=%d spin=%lld : bad=%d (future=%d worst=%d first_bad=%d)  enqueue=%.1f us/iter, "
           "total incl. drain=%.1f us/iter\n",
           mode, names[mode], N, (long long) spin, bad, forward, worst, first_bad,
           std::chrono::duration<double, std::micro>(t1 - t0).count() / N,
           std::chrono::duration<double, std::micro>(t2 - t1).count() / N);
    if (bad) {
        for (int i = 0, shown = 0; i < N && shown < 10; ++i) {
            if (h[i] != i) { printf("    out[%d]=%d\n", i, h[i]); ++shown; }
        }
    }
    return bad ? 1 : 0;
}
