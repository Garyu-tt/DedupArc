// R10: execute a *real* CPU GGML graph with the same cb_eval callback.
// This tests numeric tensor extraction, not GPT-OSS inference or agenticity.
#define main zipburger_r10_probe_entry_hidden_for_graph_test
#include "../gguf_eval_r10.cpp"
#undef main
#include "ggml-cpu.h"
#include <cmath>

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) return 1;
    ggml_backend_t available[] = {backend};
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        available, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!sched) {
        ggml_backend_free(backend);
        return 2;
    }
    const ggml_init_params params = {1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 3;
    ggml_tensor * left = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    ggml_tensor * right = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    ggml_set_input(left);
    ggml_set_input(right);
    ggml_tensor * sum = ggml_add(ctx, left, right);
    ggml_set_name(sum, "l_out-0");
    ggml_set_output(sum);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sum);
    Capture capture;
    capture.tokens = 1;
    capture.hidden = 4;
    ggml_backend_sched_set_eval_callback(sched, cb_eval, &capture);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) return 4;
    const float a[4] = {1, 2, 3, 4};
    const float b[4] = {5, 6, 7, 8};
    ggml_backend_tensor_set(left, a, 0, sizeof(a));
    ggml_backend_tensor_set(right, b, 0, sizeof(b));
    const ggml_status rc = ggml_backend_sched_graph_compute(sched, graph);
    if (rc != GGML_STATUS_SUCCESS || capture.failed) return 5;
    const auto & observed = capture.layers[0].output;
    if (observed.size() != 4) return 6;
    for (int i = 0; i < 4; i++) {
        if (std::fabs(observed[i] - (a[i] + b[i])) > 1.e-6f) return 7;
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::puts("R10 actual CPU GGML graph callback numeric extraction: PASS");
    return 0;
}
