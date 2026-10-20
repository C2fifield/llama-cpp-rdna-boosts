// WIP probe (TODO #52 part C): unified-KV cell-*set* dependence.
//
// Unlike test-multiseq-offset (a fixed "companion fill then target" shape), this
// probe takes an explicit run script so the same target token stream can be placed
// in different physical KV cell sets:
//
//   MSQS_RUNS="1:64,0:256,1:64,0:256"   -> process seq1 64 tok, seq0 256 tok, ...
//   MSQS_STEPS=N                        -> then N decode steps
//   MSQS_STEP_SEQS="0,1"                -> which sequences contribute one token per step
//
// The last run/step carries the output flag on seq0's last token, whose logits are
// hashed with FNV-1a over the whole vocab (bitwise).  Any hash difference between
// two scripts that feed seq0 the SAME token stream is a pure function of the
// physical cell set (the companion is always fully masked out of seq0's attention).
//
// Env: MSQS_RUNS, MSQS_STEPS, MSQS_STEP_SEQS, MSQS_SEQ (n_seq_max, default 2),
//      MSQS_UB (n_ubatch, default 512), MSQS_UNI (kv_unified, default 1),
//      MSQS_CTX (default 4096), MSQS_FILL (companion token id, default 1),
//      MSQS_TAG (label printed in the result line).
#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint32_t g_seq = 0; // sequence whose logits are reported

static uint64_t hash_logits(const float * l, int n_vocab, int64_t * argmax, float * vmax) {
    uint64_t h = 1469598103934665603ull;
    int64_t am = 0;
    float mv = l[0];
    for (int k = 0; k < n_vocab; ++k) {
        uint32_t b;
        memcpy(&b, &l[k], sizeof(b));
        h = (h ^ b) * 1099511628211ull;
        if (l[k] > mv) { mv = l[k]; am = k; }
    }
    *argmax = am;
    *vmax   = mv;
    return h;
}

int main(int argc, char ** argv) {
    setlocale(LC_NUMERIC, "C");
    common_params params;
    params.sampling.seed = 1234;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) { return 1; }
    ggml_backend_load_all();
    common_init_result_ptr init = common_init_from_params(params);
    llama_model * model = init->model();
    if (!model) { fprintf(stderr, "no model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const char * runs_s   = getenv("MSQS_RUNS");
    const uint32_t n_steps = getenv("MSQS_STEPS") ? (uint32_t) atoi(getenv("MSQS_STEPS")) : 0;
    const char * step_s   = getenv("MSQS_STEP_SEQS") ? getenv("MSQS_STEP_SEQS") : "0,1";
    const uint32_t n_seq  = getenv("MSQS_SEQ")  ? (uint32_t) atoi(getenv("MSQS_SEQ"))  : 2;
    const uint32_t n_ub   = getenv("MSQS_UB")   ? (uint32_t) atoi(getenv("MSQS_UB"))   : 512;
    const uint32_t n_ctx  = getenv("MSQS_CTX")  ? (uint32_t) atoi(getenv("MSQS_CTX"))  : 4096;
    const bool     uni    = getenv("MSQS_UNI")  ? atoi(getenv("MSQS_UNI")) != 0 : true;
    const int      fill   = getenv("MSQS_FILL") ? atoi(getenv("MSQS_FILL")) : 1;
    const char *   tag    = getenv("MSQS_TAG")  ? getenv("MSQS_TAG") : "-";
    g_seq = getenv("MSQS_SEQ_REPORT") ? (uint32_t) atoi(getenv("MSQS_SEQ_REPORT")) : 0;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seq;
    cparams.n_ctx      = n_ctx;
    cparams.n_batch    = n_ctx;
    cparams.n_ubatch   = n_ub;
    cparams.kv_unified = uni;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    // deterministic per-sequence token stream
    auto tok_of = [&](uint32_t s, uint32_t k) -> llama_token {
        if (s == 1) { return (llama_token) fill; }
        return (llama_token) (((int64_t) (k*7 + 11) + 13*(int64_t) s) % n_vocab);
    };
    std::vector<uint32_t> pos(n_seq, 0);

    std::vector<std::pair<uint32_t, uint32_t>> runs; // seq, n_tokens
    if (runs_s) {
        std::string s(runs_s);
        size_t p = 0;
        while (p < s.size()) {
            const size_t e = s.find(',', p);
            const std::string tok = s.substr(p, e == std::string::npos ? std::string::npos : e - p);
            const size_t c = tok.find(':');
            if (c != std::string::npos) {
                runs.emplace_back((uint32_t) atoi(tok.substr(0, c).c_str()),
                                  (uint32_t) atoi(tok.substr(c + 1).c_str()));
            }
            if (e == std::string::npos) break;
            p = e + 1;
        }
    }

    const float * last = nullptr;
    int32_t last_idx = -1;

    for (size_t r = 0; r < runs.size(); ++r) {
        const uint32_t s = runs[r].first;
        const uint32_t n = runs[r].second;
        if (s >= n_seq) { fprintf(stderr, "bad seq in run\n"); return 1; }
        // output on the last token of the last run for the report sequence
        const bool is_last_run = (r + 1 == runs.size()) && n_steps == 0;
        common_batch b(ctx);
        for (uint32_t k = 0; k < n; ++k) {
            const bool out = is_last_run && s == g_seq && k + 1 == n;
            b.add(tok_of(s, pos[s] + k), (llama_pos) (pos[s] + k), (llama_seq_id) s, out);
        }
        pos[s] += n;
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) {
            fprintf(stderr, "run %zu failed\n", r); llama_free(ctx); return 1;
        }
        if (is_last_run) {
            last = llama_get_logits_ith(ctx, b.size() - 1);
        }
    }

    if (n_steps > 0) {
        std::vector<uint32_t> step_seqs;
        {
            std::string s(step_s);
            size_t p = 0;
            while (p < s.size()) {
                const size_t e = s.find(',', p);
                step_seqs.push_back((uint32_t) atoi(s.substr(p, e == std::string::npos ? std::string::npos : e - p).c_str()));
                if (e == std::string::npos) break;
                p = e + 1;
            }
        }
        for (uint32_t st = 0; st < n_steps; ++st) {
            common_batch b(ctx);
            int32_t out_idx = -1;
            for (size_t i = 0; i < step_seqs.size(); ++i) {
                const uint32_t s = step_seqs[i];
                const int32_t idx = b.add(tok_of(s, pos[s]), (llama_pos) pos[s], (llama_seq_id) s, true);
                pos[s] += 1;
                if (s == g_seq) { last_idx = idx; }
            }
            (void) out_idx;
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) {
                fprintf(stderr, "step %u failed\n", st); llama_free(ctx); return 1;
            }
            if (st + 1 == n_steps) {
                last = llama_get_logits_ith(ctx, last_idx);
            }
        }
    }

    int64_t am = 0;
    float mv = 0.f;
    if (!last) {
        fprintf(stderr, "RUNS tag=%s: no logits (empty script)\n", tag);
        return 1;
    }
    const uint64_t h = hash_logits(last, n_vocab, &am, &mv);
    fprintf(stderr, "RUNS tag=%-28s runs=%-28s steps=%u/%s argmax=%7lld max=%.6f hash=%016llx\n",
            tag, runs_s ? runs_s : "-", n_steps, step_s, (long long) am, mv, (unsigned long long) h);

    llama_free(ctx);
    return 0;
}
