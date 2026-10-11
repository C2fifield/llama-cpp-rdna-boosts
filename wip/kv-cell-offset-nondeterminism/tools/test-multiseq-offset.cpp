// WIP probe for TODO #52 part C / issue #134: unified-KV cell-offset dependence.
//
// One unified context with up to two sequences.  The companion (seq1) is prefilled
// with MSQS_BFILL tokens *first*, which shifts the physical cell range that the
// target sequence (seq0) is allocated.  seq0's token stream is identical for every
// BFILL, so any change in seq0's logits is a pure function of the cell offset.
//
// Modes:
//   MSQS_MODE=only0 (default): prefill seq1 (BFILL tokens), then seq0 (MSQS_TOK)
//   MSQS_MODE=solo           : seq0 alone in a fresh context (reference)
//   MSQS_MODE=both           : seq0 and seq1 in one batch (seq0 first)
//
// Env: MSQS_SEQ (n_seq_max for both=2), MSQS_TOK (seq0 tokens, default 512),
//      MSQS_BFILL (seq1 tokens, default 0), MSQS_UB (n_ubatch), MSQS_UNI (kv_unified),
//      MSQS_AT (token index to report, 0 = last), MSQS_FILL (seq1 token id, default 1).
#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

static uint32_t g_at = 0;

static llama_context * make_ctx(const common_params & params, llama_model * model,
        uint32_t n_seq_max, uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch, bool unified) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seq_max;
    cparams.n_ctx      = n_ctx;
    cparams.n_batch    = n_batch;
    cparams.n_ubatch   = n_ubatch;
    cparams.kv_unified = unified;
    return llama_init_from_model(model, cparams);
}

static void report(const char * mode, uint32_t bfill, const float * l, int n_vocab) {
    if (!l) { fprintf(stderr, "OFFSET mode=%s bfill=%u: no logits\n", mode, bfill); return; }
    uint64_t h = 1469598103934665603ull;
    int64_t am = 0; float mv = l[0];
    for (int k = 0; k < n_vocab; ++k) {
        uint32_t b; memcpy(&b, &l[k], sizeof(b));
        h = (h ^ b) * 1099511628211ull;
        if (l[k] > mv) { mv = l[k]; am = k; }
    }
    fprintf(stderr, "OFFSET mode=%-5s bfill=%4u argmax=%7lld max=%.6f hash=%016llx\n",
            mode, bfill, (long long) am, mv, (unsigned long long) h);
}

static void run(const common_params & params, llama_model * model, const llama_vocab * vocab,
        const char * mode, uint32_t n_tok, uint32_t bfill, uint32_t n_ubatch, bool unified, int fill) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t n_pre = getenv("MSQS_PRE") ? (uint32_t) atoi(getenv("MSQS_PRE")) : 0;
    const uint32_t n_dec = getenv("MSQS_DEC") ? (uint32_t) atoi(getenv("MSQS_DEC")) : 0;

    // deterministic per-position token stream; seq0 and seq1 use distinct streams
    auto tok_of = [&](uint32_t s, uint32_t k) -> llama_token {
        return (llama_token) (((int64_t) (k*7 + 11) + 13*(int64_t) s) % n_vocab);
    };

    const bool both  = strcmp(mode, "both")  == 0;
    const bool solo  = strcmp(mode, "solo")  == 0;

    llama_context * ctx = make_ctx(params, model, solo ? 1 : 2, 4096, 4096, n_ubatch, unified);
    if (!ctx) { fprintf(stderr, "OFFSET mode=%s: ctx init failed\n", mode); return; }

    if (!solo && !both && bfill > 0) {
        // companion prefill first: shifts seq0's physical cell range
        common_batch b(ctx);
        for (uint32_t k = 0; k < bfill; ++k) {
            b.add((llama_token) fill, (llama_pos) k, 1, false);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) {
            fprintf(stderr, "OFFSET: companion prefill failed\n"); llama_free(ctx); return;
        }
    }

    common_batch b(ctx);
    if (n_pre > 0) {
        // prefill seq0 to n_pre tokens, then decode n_dec tokens in ONE batch (exercises the
        // decode/verify band, where the KV split is round-robin over absolute iterations)
        common_batch pb(ctx);
        for (uint32_t k = 0; k < n_pre; ++k) {
            pb.add(tok_of(0, k), (llama_pos) k, 0, false);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, pb.get()) != 0) {
            fprintf(stderr, "OFFSET: seq0 prefill failed\n"); llama_free(ctx); return;
        }
        const uint32_t n_steps = getenv("MSQS_STEPS") ? (uint32_t) atoi(getenv("MSQS_STEPS")) : 0;
        if (n_steps > 0) {
            // server-like: both sequences decode in the SAME batch, many steps; seq1 starts a
            // fresh stream at position `bfill`.  This interleaves the two sequences' cell
            // allocations across steps, which a one-shot probe cannot reproduce.
            const float * last = nullptr;
            const bool alone = getenv("MSQS_ALONE") && atoi(getenv("MSQS_ALONE")) != 0;
            for (uint32_t st = 0; st < n_steps; ++st) {
                common_batch sb(ctx);
                sb.add(tok_of(0, n_pre + st), (llama_pos) (n_pre + st), 0, st + 1 == n_steps);
                if (!alone) {
                    sb.add(tok_of(1, bfill + st), (llama_pos) (bfill + st), 1, false);
                }
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, sb.get()) != 0) {
                    fprintf(stderr, "OFFSET: interleaved step failed\n"); llama_free(ctx); return;
                }
                if (st + 1 == n_steps) {
                    last = llama_get_logits_ith(ctx, 0);
                }
            }
            report(mode, bfill, last, n_vocab);
            llama_free(ctx);
            return;
        }
        for (uint32_t k = 0; k < n_dec; ++k) {
            b.add(tok_of(0, n_pre + k), (llama_pos) (n_pre + k), 0, k == n_dec - 1);
        }
    } else {
        for (uint32_t k = 0; k < n_tok; ++k) {
            b.add(tok_of(0, k), (llama_pos) k, 0, k == (g_at ? g_at : n_tok - 1));
        }
    }
    if (both) {
        for (uint32_t k = 0; k < n_tok; ++k) {
            b.add(tok_of(1, k), (llama_pos) k, 1, false);
        }
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) {
        fprintf(stderr, "OFFSET: seq0 decode failed\n"); llama_free(ctx); return;
    }

    // only seq0's last token requested output; llama_get_logits_ith takes the batch
    // token index, so use that index directly.
    const int32_t out_i = (int32_t) (g_at ? g_at : (n_pre > 0 ? n_dec - 1 : n_tok - 1));
    report(mode, bfill, llama_get_logits_ith(ctx, out_i), n_vocab);

    llama_free(ctx);
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

    const char * mode_env = getenv("MSQS_MODE");
    const char * mode = mode_env ? mode_env : "only0";
    const uint32_t n_tok    = getenv("MSQS_TOK")   ? (uint32_t) atoi(getenv("MSQS_TOK"))   : 512;
    const uint32_t bfill    = getenv("MSQS_BFILL") ? (uint32_t) atoi(getenv("MSQS_BFILL")) : 0;
    const uint32_t n_ubatch = getenv("MSQS_UB")    ? (uint32_t) atoi(getenv("MSQS_UB"))    : 512;
    const bool     uni      = getenv("MSQS_UNI")   ? atoi(getenv("MSQS_UNI")) != 0 : true;
    const int      fill     = getenv("MSQS_FILL")  ? atoi(getenv("MSQS_FILL")) : 1;
    g_at = getenv("MSQS_AT") ? (uint32_t) atoi(getenv("MSQS_AT")) : 0;

    // MSQS_BFILL_LIST="0,32,64,...": sweep the companion fill in one process (amortizes model load)
    if (const char * list = getenv("MSQS_BFILL_LIST")) {
        std::string s(list);
        size_t p = 0;
        while (p <= s.size()) {
            const size_t e = s.find(',', p);
            const std::string tok = s.substr(p, e == std::string::npos ? std::string::npos : e - p);
            if (!tok.empty()) {
                run(params, model, vocab, mode, n_tok, (uint32_t) atoi(tok.c_str()), n_ubatch, uni, fill);
            }
            if (e == std::string::npos) break;
            p = e + 1;
        }
        return 0;
    }

    run(params, model, vocab, mode, n_tok, bfill, n_ubatch, uni, fill);
    return 0;
}
