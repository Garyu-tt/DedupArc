// Standalone deterministic unit check of GGML scheduler callback subscription.
// No GGUF or model loaded; this verifies 'ask' semantics and 24-layer bounds.
#define main zipburger_r10_native_cli_not_called_in_unit_tests
#include "../gguf_eval_r10.cpp"
#undef main

#include <cassert>
#include <cstdio>
#include <cstring>

static bool sample(const char * name, bool ask) {
    Capture data;
    ggml_tensor observed = {};
    std::snprintf(observed.name, sizeof(observed.name), "%s", name);
    return cb_eval(&observed, ask, &data);
}

int main() {
    // For ask=true return true ONLY for explicitly selected 24-layer targets.
    assert(sample("l_out-0", true));
    assert(sample("l_out-23", true));
    assert(sample("attn_post_norm-7", true));
    assert(sample("ffn_moe_topk-12", true));
    assert(sample("ffn_moe_logits_biased-4", true));
    assert(sample("ffn_moe_weights_softmax-19", true));
    assert(!sample("l_out-24", true));
    assert(!sample("l_out--1", true));
    assert(!sample("l_out-99999", true));
    assert(!sample("ffn_moe_topk-99999", true));
    assert(!sample("ffn_moe_topk-x", true));
    assert(!sample("ffn_moe_topk", true));
    assert(!sample("attn_norm-8", true));
    assert(!sample("result_output", true));

    // After-phase returns true for unobserved graph nodes and does not cancel
    // the real graph. Selected ones without buffer abort (fail-closed).
    assert(sample("result_output", false));
    assert(!sample("l_out-0", false));

    std::puts("R10 GGML callback subscription: PASS");
}
