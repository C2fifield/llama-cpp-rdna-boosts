// Hold a fixed amount of VRAM on every visible HIP device, so a 32 GiB card presents the
// free-memory figure of a 24 GiB card.  Used to reproduce a VRAM-pressure startup failure.
//   vram_fill <MiB per device> [seconds]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); return 1; } } while (0)

int main(int argc, char ** argv) {
    const size_t mib  = argc > 1 ? (size_t) atoll(argv[1]) : 8192;
    const int    secs = argc > 2 ? atoi(argv[2]) : 0; // 0 = until killed
    int n = 0;
    CK(hipGetDeviceCount(&n));
    printf("vram_fill: %d devices, %.0f MiB each\n", n, (double) mib);
    for (int i = 0; i < n; i++) {
        CK(hipSetDevice(i));
        void * p = nullptr;
        CK(hipMalloc(&p, mib * 1024 * 1024));
        // touch it so the pages are really resident
        CK(hipMemset(p, 0, mib * 1024 * 1024));
        size_t free_b = 0, total_b = 0;
        CK(hipMemGetInfo(&free_b, &total_b));
        printf("vram_fill: dev %d held %.0f MiB, free now %.0f MiB of %.0f MiB\n",
               i, (double) mib, (double) free_b / 1048576.0, (double) total_b / 1048576.0);
    }
    fflush(stdout);
    if (secs > 0) { sleep(secs); return 0; }
    for (;;) { pause(); }
}
