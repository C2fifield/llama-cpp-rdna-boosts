// pairbench: re-measure the qwen4exp ssm_alpha + ssm_beta pair on r53.
//   config "pair"    : two MUL_MATs  wb[K,48] x cur[K,N] and wa[K,48] x cur[K,N]
//   config "stacked" : one  MUL_MAT   w [K,96] x cur[K,N]
// Also compares the numerics of the two paths (pair results vs the stacked halves).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void fill(std::vector<float> & v, uint32_t seed) {
    for (size_t i = 0; i < v.size(); i++) {
        seed = seed * 1664525u + 1013904223u;
        v[i] = (float)((int)(seed >> 8) % 2001 - 1000) / 1000.0f;
    }
}

int main(int argc, char ** argv) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "no cuda backend\n"); return 1; }

    const int64_t K = 2560;
    const int64_t M = 48;
    std::vector<int64_t> ns = {256, 512, 1024, 2048, 4096, 8192};
    if (argc > 1) { ns.clear(); for (int i = 1; i < argc; i++) ns.push_back(atoll(argv[i])); }

    for (int64_t N : ns) {
        // ---- allocate operands on the device ----
        ggml_context * wctx = ggml_init({ 256*1024*1024, nullptr, true });
        ggml_tensor * wb   = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, M);
        ggml_tensor * wa   = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, M);
        ggml_tensor * ws   = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, 2*M);
        ggml_tensor * cur  = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, N);
        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);

        std::vector<float> hb(K*M), ha(K*M), hs(K*2*M), hc(K*N);
        fill(hb, 1); fill(ha, 2); fill(hc, 3);
        for (int64_t j = 0; j < M; j++) {
            for (int64_t i = 0; i < K; i++) {
                hs[j*K + i]          = hb[j*K + i];
                hs[(M+j)*K + i]      = ha[j*K + i];
            }
        }
        ggml_backend_tensor_set(wb,  hb.data(), 0, hb.size()*4);
        ggml_backend_tensor_set(wa,  ha.data(), 0, ha.size()*4);
        ggml_backend_tensor_set(ws,  hs.data(), 0, hs.size()*4);
        ggml_backend_tensor_set(cur, hc.data(), 0, hc.size()*4);

        // outputs (keep both paths separate)
        ggml_context * octx = ggml_init({ 256*1024*1024, nullptr, true });
        ggml_tensor * ob = ggml_mul_mat(octx, wb, cur);   // [M, N]
        ggml_tensor * oa = ggml_mul_mat(octx, wa, cur);   // [M, N]
        ggml_tensor * os = ggml_mul_mat(octx, ws, cur);   // [2M, N]
        ggml_backend_buffer_t obuf = ggml_backend_alloc_ctx_tensors(octx, backend);

        // ---- time pair ----
        ggml_cgraph * gpair = ggml_new_graph(octx);
        ggml_build_forward_expand(gpair, ob);
        ggml_build_forward_expand(gpair, oa);
        // ---- time stacked ----
        ggml_cgraph * gstack = ggml_new_graph(octx);
        ggml_build_forward_expand(gstack, os);

        auto time_graph = [&](ggml_cgraph * gf) -> double {
            for (int i = 0; i < 5; i++) ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            const int iters = 50;
            const double t0 = now_ms();
            for (int i = 0; i < iters; i++) ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            return (now_ms() - t0) / iters;
        };
        const double t_pair  = time_graph(gpair);
        const double t_stack = time_graph(gstack);

        // ---- numerics ----
        std::vector<float> rbp(M*N), rap(M*N), rsp(2*M*N);
        ggml_backend_tensor_get(ob, rbp.data(), 0, rbp.size()*4);
        ggml_backend_tensor_get(oa, rap.data(), 0, rap.size()*4);
        ggml_backend_tensor_get(os, rsp.data(), 0, rsp.size()*4);
        double max_b = 0.0, max_a = 0.0;
        for (int64_t n = 0; n < N; n++) {
            for (int64_t m = 0; m < M; m++) {
                max_b = std::max(max_b, (double) std::fabs(rbp[n*M+m] - rsp[n*2*M + m]));
                max_a = std::max(max_a, (double) std::fabs(rap[n*M+m] - rsp[n*2*M + M + m]));
            }
        }
        const double wbytes = (double) K*M*4*2; // both paths read the same weight bytes
        fprintf(stderr,
            "N=%5lld  pair %8.3f ms  stacked %8.3f ms  ratio %.3fx  GB/s pair %6.1f stack %6.1f  maxdiff b=%.2e a=%.2e\n",
            (long long) N, t_pair, t_stack, t_pair/t_stack,
            wbytes/1e9 / (t_pair/1000.0), wbytes/1e9 / (t_stack/1000.0), max_b, max_a);

        ggml_backend_buffer_free(wbuf);
        ggml_backend_buffer_free(obuf);
        ggml_free(octx);
        ggml_free(wctx);
    }

    ggml_backend_free(backend);
    return 0;
}
