// Standalone reproducer for the wip/async-input-graph-ordering defect.
//
// Question: does a `hipStreamWaitEvent(compute, ev)` issued on the compute stream immediately before a
// *captured graph* launch order the replay after a copy recorded on another stream (here the per-thread
// default stream, as in ggml_cuda_async_input_*)?
//
// Model: the input buffer `dst` is overwritten each iteration by an async H2D copy from a pinned slot
// (a fresh slot per iteration, exactly like the 16 MiB ring), an event is recorded after the copy, the
// compute stream waits on it, then the graph (or the kernel directly) reads `dst[0]` and writes
// out[atomicAdd(ctr)] = dst[0].
//
// After N iterations out[i] must equal i.  A stale read shows up as out[i] != i (typically i+k).
//
// modes:
//   0  direct kernel launch on the compute stream (no graph)
//   1  graph replay (wait -> graphLaunch)
//   2  graph replay with a no-op kernel launch between the wait and the launch
//   3  graph replay, upload issued on an explicit non-blocking stream (not hipStreamPerThread)
//   4  graph replay, but the copy is issued on the *compute* stream (wait is pointless; control)
//   5  graph replay with the delivery's done_ev guard (copy waits for the previous graph's completion event)
//   6  graph replay, done_ev guard only (no d.ev wait)
//   7  direct kernel with the done_ev guard (does the guard work without graphs?)
//   8  graph + done_ev guard, plus a host hipEventSynchronize(done_ev) before the copy
//   9  graph + done_ev guard, plus a host hipStreamSynchronize(compute) before the copy (fully serialized)
//
// build:
//   /opt/rocm-10.1.0-gfx120X/lib/llvm/bin/clang++ -x hip --offload-arch=gfx1201 -O2 \
//     -isystem/opt/rocm-10.1.0-gfx120X/include -o ain_hip_test ain_hip_test.cpp
// run:
//   ./ain_hip_test <mode> [iterations] [threads_per_block]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static constexpr int B = 32;

__global__ void k_read(const int * in, int * out, int * ctr) {
    const int i = atomicAdd(ctr, 1);
    out[i] = in[0];
}

__global__ void k_noop(int * out) {
    if (blockIdx.x == 0xffffffffu) { out[0] = 1; }
}

int main(int argc, char ** argv) {
    const int mode = argc > 1 ? atoi(argv[1]) : 1;
    const int N    = argc > 2 ? atoi(argv[2]) : 4000;
    const int thr  = argc > 3 ? atoi(argv[3]) : 64;

    hipStream_t compute, upload_expl = nullptr;
    HC(hipStreamCreateWithFlags(&compute, hipStreamNonBlocking));
    if (mode == 3) HC(hipStreamCreateWithFlags(&upload_expl, hipStreamNonBlocking));
    const hipStream_t upload = mode == 3 ? upload_expl : hipStreamPerThread;

    int * dst = nullptr, * out = nullptr, * ctr = nullptr;
    HC(hipMalloc(&dst, B * sizeof(int)));
    HC(hipMalloc(&out, N * sizeof(int)));
    HC(hipMalloc(&ctr, sizeof(int)));
    HC(hipMemset(dst, 0, B * sizeof(int)));
    HC(hipMemset(out, -1, N * sizeof(int)));
    HC(hipMemset(ctr, 0, sizeof(int)));

    // N pinned slots, one per iteration (like the ring: never rewritten)
    std::vector<int *> slots(N);
    for (int i = 0; i < N; ++i) {
        HC(hipHostMalloc((void **) &slots[i], B * sizeof(int)));
        for (int j = 0; j < B; ++j) slots[i][j] = i;
    }

    hipEvent_t ev, done_ev;
    HC(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
    HC(hipEventCreateWithFlags(&done_ev, hipEventDisableTiming));
    const bool guard = mode >= 5;

    hipGraph_t     graph = nullptr;
    hipGraphExec_t exec  = nullptr;
    if (mode != 0) {
        HC(hipStreamBeginCapture(compute, hipStreamCaptureModeRelaxed));
        k_read<<<1, thr, 0, compute>>>(dst, out, ctr);
        HC(hipStreamEndCapture(compute, &graph));
        HC(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }

    for (int i = 0; i < N; ++i) {
        if (guard && i > 0) {
            // the delivery's copy_locked(): the next upload may not land before the previous graph finished
            if (mode == 8)      { HC(hipEventRecord(done_ev, compute)); HC(hipEventSynchronize(done_ev)); }
            else if (mode == 9) { HC(hipStreamSynchronize(compute)); }
            else                { HC(hipStreamWaitEvent(upload, done_ev, 0)); }
        }
        if (mode == 4) {
            HC(hipMemcpyAsync(dst, slots[i], B * sizeof(int), hipMemcpyHostToDevice, compute));
            HC(hipEventRecord(ev, compute));
            HC(hipStreamWaitEvent(compute, ev, 0));
        } else {
            HC(hipMemcpyAsync(dst, slots[i], B * sizeof(int), hipMemcpyHostToDevice, upload));
            if (mode != 6) {
                HC(hipEventRecord(ev, upload));
                HC(hipStreamWaitEvent(compute, ev, 0));
            }
        }
        if (mode == 0 || mode == 7) {
            k_read<<<1, thr, 0, compute>>>(dst, out, ctr);
        } else if (mode == 2) {
            k_noop<<<1, 1, 0, compute>>>(out);
            HC(hipGraphLaunch(exec, compute));
        } else {
            HC(hipGraphLaunch(exec, compute));
        }
        if (mode == 5 || mode == 6 || mode == 8 || mode == 9) {
            HC(hipEventRecord(done_ev, compute));
        }
    }

    HC(hipStreamSynchronize(compute));
    HC(hipStreamSynchronize(hipStreamPerThread));
    HC(hipDeviceSynchronize());

    std::vector<int> h(out ? N : 0);
    HC(hipMemcpy(h.data(), out, N * sizeof(int), hipMemcpyDeviceToHost));

    int bad = 0, forward_gap = 0, first_bad = -1, worst_gap = 0;
    for (int i = 0; i < N; ++i) {
        if (h[i] != i) {
            ++bad;
            if (first_bad < 0) first_bad = i;
            const int gap = h[i] - i;           // >0: read a value uploaded later than iteration i
            if (gap > 0) ++forward_gap;
            if (gap > worst_gap) worst_gap = gap;
        }
    }
    const char * names[] = { "direct-kernel", "graph-replay", "graph+noop", "graph/explicit-stream", "graph/copy-on-compute",
                             "graph+done_ev", "graph+done_ev(no ev wait)", "direct+done_ev", "graph+done_ev+hostsync", "graph+done_ev+serialized" };
    printf("mode=%d (%s) N=%d thr=%d : bad=%d (read-a-future-value=%d, worst gap=%d, first_bad=%d)\n",
           mode, names[mode], N, thr, bad, forward_gap, worst_gap, first_bad);
    if (bad) {
        for (int i = 0, shown = 0; i < N && shown < 8; ++i) {
            if (h[i] != i) { printf("    out[%d]=%d\n", i, h[i]); ++shown; }
        }
    }
    return bad ? 1 : 0;
}
