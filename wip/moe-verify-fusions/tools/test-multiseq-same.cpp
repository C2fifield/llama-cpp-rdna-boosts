// WIP probe (TODO #52 B): four sequences, identical tokens, one decode.
// All sequences must produce identical logits; if seq1..3 differ from seq0 the
// single-sequence-per-ubatch split corrupts the shared forward or the cache state.
#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint32_t g_at     = 0; // token index to report (0 = last)
static uint32_t g_ref_ub = 0; // reference n_ubatch (0 = same as n_ubatch)

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

static void run(const common_params & params, llama_model * model, llama_context * ctx_tok,
        const llama_vocab * vocab, uint32_t n_seq, uint32_t n_tok, uint32_t n_ubatch,
        bool unified, bool split_calls) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> base = common_tokenize(ctx_tok, "The quick brown fox jumps over the lazy dog", true);
    if (base.empty()) { base = { 1, 2, 3, 4, 5, 6, 7, 8, 9 }; }
    while (base.size() < n_tok) { base.push_back(base[base.size() % 9]); }
    base.resize(n_tok);
    auto tok_of = [&](uint32_t s, uint32_t k) -> llama_token {
        return (llama_token) (((int64_t) base[k] + 13*(int64_t) s) % n_vocab);
    };

    auto report = [&](uint32_t s, const float * l) {
        if (!l) { fprintf(stderr, "MSQSAME split=%d seq=%u: no logits\n", (int) split_calls, s); return; }
        uint64_t h = 1469598103934665603ull;
        int64_t am = 0; float mv = l[0];
        for (int k = 0; k < n_vocab; ++k) {
            uint32_t b; memcpy(&b, &l[k], sizeof(b));
            h = (h ^ b) * 1099511628211ull;
            if (l[k] > mv) { mv = l[k]; am = k; }
        }
        fprintf(stderr, "MSQSAME split=%d uni=%d seq=%u argmax=%lld max=%.6f hash=%016llx\n",
                (int) split_calls, (int) unified, s, (long long) am, mv, (unsigned long long) h);
    };

    if (split_calls) {
        // ground truth: one fresh context per sequence
        for (uint32_t s = 0; s < n_seq; ++s) {
            llama_context * ctx = make_ctx(params, model, 1, 2048, 2048, g_ref_ub ? g_ref_ub : n_ubatch, unified);
            if (!ctx) { fprintf(stderr, "ctx init failed\n"); return; }
            common_batch batch(ctx);
            for (uint32_t k = 0; k < n_tok; ++k) {
                batch.add(tok_of(s, k), (llama_pos) k, 0, true);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                fprintf(stderr, "decode failed seq %u\n", s); llama_free(ctx); return;
            }
            report(s, llama_get_logits_ith(ctx, (int32_t) (g_at ? g_at : n_tok - 1)));
            llama_free(ctx);
        }
    } else {
        llama_context * ctx = make_ctx(params, model, n_seq, 2048, 2048, n_ubatch, unified);
        if (!ctx) { fprintf(stderr, "ctx init failed\n"); return; }
        common_batch batch(ctx);
        for (uint32_t s = 0; s < n_seq; ++s) {
            for (uint32_t k = 0; k < n_tok; ++k) {
                batch.add(tok_of(s, k), (llama_pos) k, (llama_seq_id) s, true);
            }
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "decode failed\n"); llama_free(ctx); return;
        }
        for (uint32_t s = 0; s < n_seq; ++s) {
            report(s, llama_get_logits_ith(ctx, (int32_t) (s*n_tok + (g_at ? g_at : n_tok - 1))));
        }
        llama_free(ctx);
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    params.sampling.seed = 1234;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) { return 1; }
    ggml_backend_load_all();
    common_init_result_ptr init = common_init_from_params(params);
    llama_model * model = init->model();
    if (!model) { fprintf(stderr, "no model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const uint32_t n_seq    = getenv("MSQS_SEQ")  ? (uint32_t) atoi(getenv("MSQS_SEQ"))  : 4;
    const uint32_t n_tok    = getenv("MSQS_TOK")  ? (uint32_t) atoi(getenv("MSQS_TOK"))  : 64;
    const uint32_t n_ubatch = getenv("MSQS_UB")   ? (uint32_t) atoi(getenv("MSQS_UB"))   : 512;
    const bool uni          = getenv("MSQS_UNI")  ? atoi(getenv("MSQS_UNI")) != 0 : false;
    g_at     = getenv("MSQS_AT")     ? (uint32_t) atoi(getenv("MSQS_AT"))     : 0;
    g_ref_ub = getenv("MSQS_REF_UB") ? (uint32_t) atoi(getenv("MSQS_REF_UB")) : 0;

    run(params, model, init->context(), vocab, n_seq, n_tok, n_ubatch, uni, false);
    run(params, model, init->context(), vocab, n_seq, n_tok, n_ubatch, uni, true);
    return 0;
}
