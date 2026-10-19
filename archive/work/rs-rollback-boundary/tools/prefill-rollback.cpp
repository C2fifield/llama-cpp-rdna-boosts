// Prefill-then-rollback probe (wip/rs-rollback-boundary open question 2).
//
// The §27 invariant: "every batch that can be rolled back into must run the kernel that writes the
// snapshots it will read."  The guard's premise (issue #131) is that the whole-batch chunked GDN prefill
// writes only the NEWEST snapshot, so a rollback that lands inside a large prefill batch reads an
// unwritten plane.  Nobody has yet demonstrated a bit-level consequence for that ordering.  This probe
// does, deterministically, on one device:
//
//   prefill P tokens (in `ub`-sized batches, so the last batch is a full `ub`)
//   reference:  from the post-prefill state, decode X at pos P            -> L_ref, S_ref
//   test:       from the same state, seq_rm back k tokens (k <= n_rs_seq), re-decode the last k text
//               tokens one at a time, then decode X                     -> L_rb,  S_rb
//
// L_rb != L_ref (or S_rb != S_ref) means the restore read a state the prefill batch never wrote.
// A/compare with GGML_CUDA_GDN_CHUNKED=0 (the sequential kernel) as the exact-fallback control.
//
// usage: prefill-rollback <model> <text> <P> <ub>
// env:   K=1,2,3 (comma list) RS=3 CTX=4096 CTK/CTV=f16 NGL=99 NCMOE=-1 SPLIT=layer STATE=0 QUIET=1
#include "llama.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <string>
#include <vector>

static int argmax(const float * a, int n) { int b = 0; for (int i = 1; i < n; ++i) if (a[i] > a[b]) b = i; return b; }
static uint64_t fh(const float * a, int n) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char * p = (const unsigned char *) a;
    for (size_t i = 0; i < (size_t) n * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
static uint64_t bh(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char * b = (const unsigned char *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
static ggml_type kvt(const char * n) {
    if (!n) return GGML_TYPE_F16;
    if (!strcmp(n,"f16"))   return GGML_TYPE_F16;
    if (!strcmp(n,"q8_0"))  return GGML_TYPE_Q8_0;
    if (!strcmp(n,"bf16"))  return GGML_TYPE_BF16;
    return GGML_TYPE_F16;
}
static void quiet_log(ggml_log_level, const char *, void *) {}

// magnitude of a float-array difference: max abs and NMSE (so "rounding" and "corrupt" are separable)
static void fdiff(const float * a, const float * b, int n, double * maxabs, double * nmse) {
    double mx = 0, num = 0, den = 0;
    for (int i = 0; i < n; ++i) {
        const double d = (double) a[i] - (double) b[i];
        if (fabs(d) > mx) mx = fabs(d);
        num += d * d; den += (double) a[i] * (double) a[i];
    }
    *maxabs = mx;
    *nmse = den > 0 ? num / den : num;
}

// decode one token, logits enabled
static bool dec1(llama_context * c, llama_token t, llama_pos p, int seq) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.n_tokens = 1; b.token[0] = t; b.pos[0] = p;
    b.n_seq_id[0] = 1; b.seq_id[0][0] = seq; b.logits[0] = 1;
    const bool ok = llama_decode(c, b) == 0;
    llama_batch_free(b);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <model> <text> <P> <ub>\n", argv[0]); return 1; }
    const char * mpath = argv[1]; const char * tpath = argv[2];
    const int P = atoi(argv[3]); const int ub = atoi(argv[4]);
    const int CTX  = getenv("CTX")  ? atoi(getenv("CTX"))  : 4096;
    const int RS   = getenv("RS")   ? atoi(getenv("RS"))   : 3;
    const char * Ks = getenv("K")   ? getenv("K")          : "1,2,3";
    const ggml_type tk = kvt(getenv("CTK"));
    const ggml_type tv = getenv("CTV") ? kvt(getenv("CTV")) : tk;
    const bool do_state = getenv("STATE") ? atoi(getenv("STATE")) != 0 : true;
    if (ub < 1) { fprintf(stderr, "ub must be >= 1\n"); return 1; }

    if (getenv("QUIET")) llama_log_set(quiet_log, nullptr);
    ggml_backend_load_all(); llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = getenv("NGL") ? atoi(getenv("NGL")) : 99;
    static std::list<std::string> ov_strs;
    static std::vector<llama_model_tensor_buft_override> ov;
    const int n_cpu_moe = getenv("NCMOE") ? atoi(getenv("NCMOE")) : -1;
    if (n_cpu_moe >= 0) {
        for (int i = 0; i < n_cpu_moe; ++i) {
            ov_strs.push_back("blk\\." + std::to_string(i) + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
            ov.push_back({ ov_strs.back().c_str(), ggml_backend_cpu_buffer_type() });
        }
        ov.push_back({ nullptr, nullptr });
        mp.tensor_buft_overrides = ov.data();
    }
    if (const char * sp = getenv("SPLIT")) {
        if      (!strcmp(sp, "tensor")) mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        else if (!strcmp(sp, "row"))    mp.split_mode = LLAMA_SPLIT_MODE_ROW;
        else if (!strcmp(sp, "layer"))  mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
    }

    llama_model * model = llama_model_load_from_file(mpath, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);

    std::string text;
    { FILE * f = fopen(tpath, "rb"); if (!f) { perror("text"); return 1; }
      char b[65536]; size_t r; while ((r = fread(b, 1, sizeof b, f)) > 0) text.append(b, r); fclose(f); }
    std::vector<llama_token> toks(262144);
    int n = llama_tokenize(vocab, text.data(), (int) text.size(), toks.data(), (int) toks.size(), false, false);
    if (n < P) { fprintf(stderr, "text too short (%d < %d)\n", n, P); return 1; }

    const char * ch = getenv("GGML_CUDA_GDN_CHUNKED");
    const bool mode_ref = !getenv("MODE") || strcmp(getenv("MODE"), "ref") == 0;
    fprintf(stderr, "[prb] mode=%s P=%d ub=%d RS=%d k=%s chunked=%s ncmoe=%d\n",
            mode_ref ? "ref" : "replay", P, ub, RS, Ks, ch ? ch : "default", n_cpu_moe);

    auto mk_ctx = [&]() {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = CTX; cp.n_batch = ub; cp.n_ubatch = ub; cp.n_seq_max = 1; cp.n_rs_seq = (uint32_t) RS;
        cp.type_k = tk; cp.type_v = tv; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        return llama_init_from_model(model, cp);
    };
    // prefill tokens [0, n) in `ub`-sized batches; logits on for the last token
    auto prefill = [&](llama_context * c, int np) -> bool {
        for (int i = 0; i < np; i += ub) {
            const int m = (np - i < ub) ? (np - i) : ub;
            llama_batch b = llama_batch_init(m, 0, 1);
            for (int j = 0; j < m; ++j) {
                b.token[j] = toks[i + j]; b.pos[j] = i + j;
                b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = (i + j == np - 1) ? 1 : 0;
                b.n_tokens++;
            }
            if (llama_decode(c, b)) { llama_batch_free(b); return false; }
            llama_batch_free(b);
        }
        return true;
    };
    auto get_state = [&](llama_context * c, std::vector<uint8_t> & st) {
        st.resize(llama_state_seq_get_size(c, 0));
        if (!st.empty()) llama_state_seq_get_data(c, st.data(), st.size(), 0);
        return st;
    };

    if (mode_ref) {
        // MODE=ref (default): compare the state a rollback RESTORES against the state the same kernel
        // produces when the prefill simply ENDS at that position.  Both are "state at position P-k-1" and
        // both come from a wide prefill batch, so kernel-shape arithmetic cancels; only the snapshot the
        // rollback reads is under test.
        int bad = 0, ran = 0;
        for (const char * kp = Ks; *kp; ) {
            const int k = atoi(kp);
            while (*kp && *kp != ',') ++kp;
            if (*kp == ',') ++kp;
            if (k < 1 || k > RS || k >= P) { printf("k=%d SKIP (RS=%d)\n", k, RS); continue; }

            llama_context * ca = mk_ctx();
            llama_context * cb = mk_ctx();
            if (!ca || !cb) { fprintf(stderr, "ctx init failed\n"); return 1; }
            std::vector<uint8_t> sa, sb;
            if (!prefill(ca, P)) { fprintf(stderr, "prefill A fail\n"); return 1; }
            llama_memory_t ma = llama_get_memory(ca);
            const bool rm_ok = llama_memory_seq_rm(ma, 0, P - k, -1);
            get_state(ca, sa);
            if (!prefill(cb, P - k)) { fprintf(stderr, "prefill B fail\n"); return 1; }
            get_state(cb, sb);
            // Compare SEMANTICS, not the state blob: `llama_state_seq_get_data` also carries the KV and
            // recurrent cell layout, which legitimately differs between a context that prefilled P tokens
            // and one that prefilled P-k.  Instead replay the same k-token suffix one at a time in both
            // contexts and compare the LOGITS at each step -- same suffix, same kernel shapes, so only the
            // pre-state the rollback restored can differ.
            bool ok = rm_ok;
            std::vector<float> Lb_ref;
            for (int j = P - k; j < P; ++j) {
                if (!dec1(ca, toks[j], j, 0)) { fprintf(stderr, "A replay fail\n"); return 1; }
                if (!dec1(cb, toks[j], j, 0)) { fprintf(stderr, "B replay fail\n"); return 1; }
                const float * la = llama_get_logits_ith(ca, 0);
                const float * lb = llama_get_logits_ith(cb, 0);
                double mx = 0, nmse = 0;
                fdiff(la, lb, nv, &mx, &nmse);
                const bool same = fh(la, nv) == fh(lb, nv);
                printf("k=%d pos=%d %s maxabs=%.3e nmse=%.3e\n", k, j, same ? "OK  " : "DIFF", mx, nmse);
                if (!same) ok = false;
            }
            printf("k=%d rollback=%s  %s  (state bytes %zu vs %zu, layout not compared)\n",
                   k, rm_ok ? "ok" : "REFUSED", ok ? "PASS" : "FAIL", sa.size(), sb.size());
            ran++; if (!ok) bad++;
            llama_free(ca); llama_free(cb);
        }
        fprintf(stderr, "[prb] arms run=%d mismatches=%d\n", ran, bad);
        llama_model_free(model); llama_backend_free();
        return bad == 0 ? 0 : 2;
    }

    llama_context * c = mk_ctx();
    if (!c) { fprintf(stderr, "ctx init failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(c);
    if (!prefill(c, P)) { fprintf(stderr, "prefill fail\n"); return 1; }

    const int last_m = (P % ub == 0) ? ub : (P % ub);
    const float * lp = llama_get_logits_ith(c, -1);
    if (!lp) { fprintf(stderr, "no post-prefill logits\n"); return 1; }
    const llama_token X = (llama_token) argmax(lp, nv);
    std::vector<uint8_t> st_ref;
    if (do_state) get_state(c, st_ref);
    if (!dec1(c, X, P, 0)) { fprintf(stderr, "ref decode fail\n"); return 1; }
    std::vector<float> Lref_v(llama_get_logits_ith(c, 0), llama_get_logits_ith(c, 0) + nv);
    const uint64_t L_ref = fh(Lref_v.data(), nv);
    if (do_state) get_state(c, st_ref);

    int bad = 0, ran = 0;
    for (const char * kp = Ks; *kp; ) {
        const int k = atoi(kp);
        while (*kp && *kp != ',') ++kp;
        if (*kp == ',') ++kp;
        if (k < 1 || k > last_m || k > RS) { printf("k=%d SKIP (last batch %d, RS=%d)\n", k, last_m, RS); continue; }
        if (!llama_memory_seq_rm(mem, 0, P - k, -1)) { printf("k=%d seq_rm(%d,-1) REFUSED\n", k, P - k); continue; }
        for (int j = P - k; j < P; ++j) if (!dec1(c, toks[j], j, 0)) { fprintf(stderr, "replay fail\n"); return 1; }
        if (!dec1(c, X, P, 0)) { fprintf(stderr, "test decode fail\n"); return 1; }
        double mx = 0, nmse = 0;
        fdiff(llama_get_logits_ith(c, 0), Lref_v.data(), nv, &mx, &nmse);
        printf("k=%d logits %016llx vs %016llx  maxabs=%.3e nmse=%.3e\n", k,
               (unsigned long long) fh(llama_get_logits_ith(c, 0), nv), (unsigned long long) L_ref, mx, nmse);
        ran++;
        if (do_state) {
            if (!llama_state_seq_set_data(c, st_ref.data(), st_ref.size(), 0)) {
                fprintf(stderr, "[prb] state restore unavailable: run each K in its own process\n");
                break;
            }
        }
    }
    fprintf(stderr, "[prb] arms run=%d\n", ran);
    llama_free(c); llama_model_free(model); llama_backend_free();
    return 0;
}
