// ZipBurger R10: opt-in, read-only local GPT-OSS-20B GGUF activation probe.
// Compile against an explicitly pinned llama.cpp revision (see README_R10).
// No model conversion, network, weight rewrites, server changes or generation.
// Captures 24 l_out/attn_post_norm/topk/router scores/router weights,
// and all-token top1 predictions from a single <=8-token prompt prefill.
// Output: private ZBR10BIN little-endian research capture, never a model.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kLayers = 24;
constexpr int kExperts = 32;
constexpr int kTopk = 4;
constexpr int kMaxTokens = 8;
constexpr int kMaxHidden = 8192;
constexpr size_t kMaxPromptBytes = 2048;
constexpr char kMagic[8] = {'Z','B','R','1','0','B','I','N'};

struct Layer {
    std::vector<float> output, expert_input, route_logits, route_weights;
    std::vector<int32_t> topk;
};
struct Capture {
    int tokens = 0, hidden = 0;
    bool failed = false;
    std::string reason;
    std::array<Layer, kLayers> layers;
};
enum class Kind { skip, output, input, topk, logits, weights };

bool target(const char * name, Kind & kind, int & layer) {
    const std::array<std::pair<const char *, Kind>, 5> prefixes = {{
        {"l_out-", Kind::output},
        {"attn_post_norm-", Kind::input},
        {"ffn_moe_topk-", Kind::topk},
        {"ffn_moe_logits_biased-", Kind::logits},
        {"ffn_moe_weights_softmax-", Kind::weights},
    }};
    for (const auto & p : prefixes) {
        const size_t n = std::strlen(p.first);
        if (std::strncmp(name, p.first, n) != 0) continue;
        const char * suffix = name + n;
        if (*suffix < '0' || *suffix > '9') return false;
        unsigned value = 0;
        for (const char * q = suffix; *q; ++q) {
            if (*q < '0' || *q > '9') return false;
            value = value * 10 + static_cast<unsigned>(*q - '0');
            if (value >= kLayers) return false;
        }
        kind = p.second;
        layer = static_cast<int>(value);
        return true;
    }
    return false;
}

bool cb_eval(ggml_tensor * t, bool ask, void * data) {
    auto & capture = *static_cast<Capture *>(data);
    Kind kind = Kind::skip;
    int layer = -1;
    // ask=true: subscribe only to the five bounded labels per layer.
    // ask=false: returning false cancels the actual inference.
    if (!target(t->name, kind, layer)) return !ask;
    if (ask) return true;
    if (capture.failed) return false;
    try {
        if (!t->buffer || !ggml_is_contiguous(t)) {
            throw std::runtime_error("unallocated or non-contiguous observed tensor");
        }
        const bool is_ids = kind == Kind::topk;
        const int cols = kind == Kind::output || kind == Kind::input ? capture.hidden :
                         kind == Kind::topk || kind == Kind::weights ? kTopk : kExperts;
        const bool ok_shape = kind == Kind::weights ?
            (t->ne[0] == 1 && t->ne[1] == kTopk &&
             t->ne[2] == capture.tokens && t->ne[3] == 1) :
            (t->ne[0] == cols && t->ne[1] == capture.tokens &&
             t->ne[2] == 1 && t->ne[3] == 1);
        if (!ok_shape) {
            throw std::runtime_error("unexpected tensor shape for " + std::string(t->name));
        }
        const size_t count = static_cast<size_t>(cols) * capture.tokens;
        auto & out = capture.layers[layer];
        if (is_ids) {
            if (t->type != GGML_TYPE_I32 || !out.topk.empty()) {
                throw std::runtime_error("unsupported/duplicate router indices");
            }
            out.topk.resize(count);
            ggml_backend_tensor_get(t, out.topk.data(), 0, count * sizeof(int32_t));
            for (int32_t id : out.topk) {
                if (id < 0 || id >= kExperts) throw std::runtime_error("invalid router ID");
            }
        } else {
            std::vector<float> * dst = kind == Kind::output ? &out.output :
                kind == Kind::input ? &out.expert_input :
                kind == Kind::logits ? &out.route_logits : &out.route_weights;
            if (!dst->empty()) throw std::runtime_error("duplicate observed tensor");
            dst->resize(count);
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(t, dst->data(), 0, count * sizeof(float));
            } else if (t->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> fp16(count);
                ggml_backend_tensor_get(t, fp16.data(), 0, count * sizeof(ggml_fp16_t));
                for (size_t i = 0; i < count; ++i) (*dst)[i] = ggml_fp16_to_fp32(fp16[i]);
            } else {
                throw std::runtime_error("unsupported observed float type");
            }
            if (!std::all_of(dst->begin(), dst->end(), [](float v) {
                return v == v && v != std::numeric_limits<float>::infinity() &&
                       v != -std::numeric_limits<float>::infinity();
            })) throw std::runtime_error("NaN/Inf in observed tensor");
        }
    } catch (const std::exception & e) {
        capture.failed = true;
        capture.reason = e.what();
        return false;
    }
    return true;
}

void u32(std::ofstream & f, uint32_t n) {
    unsigned char b[4] = {
        static_cast<unsigned char>(n),
        static_cast<unsigned char>(n >> 8),
        static_cast<unsigned char>(n >> 16),
        static_cast<unsigned char>(n >> 24)
    };
    f.write(reinterpret_cast<const char *>(b), 4);
}
template<class T>
void write_vec(std::ofstream & f, const std::vector<T> & values) {
    f.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
}

std::string read_prompt(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("prompt-file unavailable");
    const std::streamoff n = f.tellg();
    if (n <= 0 || n > static_cast<std::streamoff>(kMaxPromptBytes)) {
        throw std::runtime_error("prompt file exceeds local research cap");
    }
    f.seekg(0);
    std::string prompt(static_cast<size_t>(n), '\0');
    f.read(&prompt[0], n);
    if (!f || prompt.find('\0') != std::string::npos) {
        throw std::runtime_error("prompt file truncated/binary");
    }
    return prompt;
}
void write_capture(const std::string & path, const Capture & c,
                   const std::vector<int32_t> & preds,
                   const std::vector<llama_token> & token_ids) {
    for (int i = 0; i < kLayers; ++i) {
        const auto & l = c.layers[i];
        const size_t h = static_cast<size_t>(c.tokens) * c.hidden;
        if (l.output.size() != h || l.expert_input.size() != h ||
            l.topk.size() != static_cast<size_t>(c.tokens * kTopk) ||
            l.route_logits.size() != static_cast<size_t>(c.tokens * kExperts) ||
            l.route_weights.size() != static_cast<size_t>(c.tokens * kTopk)) {
            throw std::runtime_error("mandatory l_out/MoE observation missing at layer " +
                                     std::to_string(i));
        }
    }
    if (preds.size() != static_cast<size_t>(c.tokens) ||
        token_ids.size() != static_cast<size_t>(c.tokens)) {
        throw std::runtime_error("incomplete per-token predictions");
    }
    std::ifstream present(path, std::ios::binary);
    if (present.good()) throw std::runtime_error("refusing to overwrite capture file");
    std::ofstream f(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot open output");
    f.write(kMagic, sizeof(kMagic));
    for (unsigned n : {static_cast<unsigned>(c.tokens),
                       static_cast<unsigned>(c.hidden), unsigned(kLayers),
                       unsigned(kTopk), unsigned(kExperts)}) u32(f, n);
    for (const Layer & l : c.layers) {
        write_vec(f, l.output);
        write_vec(f, l.expert_input);
        write_vec(f, l.topk);
        write_vec(f, l.route_logits);
        write_vec(f, l.route_weights);
    }
    write_vec(f, preds);
    // Token ids are potentially reversible; keep capture strictly local/private.
    for (const auto token : token_ids) u32(f, static_cast<uint32_t>(token));
    f.flush();
    if (!f) throw std::runtime_error("capture output write failed");
}
} // namespace

int main(int argc, char ** argv) {
    std::string model_file, prompt_file, output_file;
    int threads = 2;
    try {
        if (argc != 7 && argc != 9) {
            throw std::runtime_error(
                "usage: gguf_eval_r10 --model /local/model.gguf "
                "--prompt-file /private/prompt.txt --out /private/trace.zbr10 "
                "[--threads 1..16]");
        }
        for (int i = 1; i + 1 < argc; i += 2) {
            const std::string key = argv[i], value = argv[i + 1];
            if (key == "--model") model_file = value;
            else if (key == "--prompt-file") prompt_file = value;
            else if (key == "--out") output_file = value;
            else if (key == "--threads") threads = std::stoi(value);
            else throw std::runtime_error("unknown command argument");
        }
        if (model_file.empty() || prompt_file.empty() || output_file.empty() ||
            threads < 1 || threads > 16) throw std::runtime_error("invalid options");

        const std::string prompt = read_prompt(prompt_file);
        llama_backend_init();
        struct Backend { ~Backend() { llama_backend_free(); } } backend_guard;
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0; // explicitly CPU-only
        // Upstream selects its own default load/mmap mode.
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_file.c_str(), params), llama_model_free);
        if (!model || llama_model_n_layer(model.get()) != kLayers ||
            llama_model_n_embd(model.get()) < 1 ||
            llama_model_n_embd(model.get()) > kMaxHidden) {
            throw std::runtime_error("expected locally available GPT-OSS-20B 24-layer model");
        }
        char architecture[80] = {};
        if (llama_model_meta_val_str(model.get(), "general.architecture",
                                     architecture, sizeof(architecture)) < 0 ||
            std::strcmp(architecture, "gpt-oss") != 0) {
            throw std::runtime_error("GGUF architecture is not gpt-oss");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        std::vector<llama_token> tokens(kMaxTokens);
        int n = llama_tokenize(vocab, prompt.c_str(), static_cast<int>(prompt.size()),
                               tokens.data(), kMaxTokens, true, true);
        if (n < 1 || n > kMaxTokens) throw std::runtime_error("prompt exceeds 8-token prefill limit");
        tokens.resize(n);

        Capture capture;
        capture.tokens = n;
        capture.hidden = llama_model_n_embd(model.get());
        auto cp = llama_context_default_params();
        cp.n_ctx = 64;
        cp.n_batch = 32;
        cp.n_ubatch = 32;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.cb_eval = cb_eval;
        cp.cb_eval_user_data = &capture;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("cannot create probe context");

        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
            llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        if (!batch) throw std::runtime_error("cannot allocate probe batch");
        for (int i = 0; i < n; ++i) {
            int32_t j = llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            llama_pos pos = i;
            if (j < 0 || !llama_batch_ext_set_pos(batch.get(), j, &pos) ||
                !llama_batch_ext_set_output_logits(batch.get(), j, true)) {
                throw std::runtime_error("cannot populate probe batch");
            }
        }
        const int code = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (code != 0 || capture.failed) {
            throw std::runtime_error("inference/callback failed: " + capture.reason);
        }
        std::vector<int32_t> top1;
        const int32_t nvocab = llama_vocab_n_tokens(vocab);
        if (nvocab < 1) throw std::runtime_error("invalid vocab size");
        for (int i = 0; i < n; ++i) {
            const float * logits = llama_get_logits_ith(ctx.get(), i);
            if (!logits) throw std::runtime_error("missing output logits");
            auto best = std::max_element(logits, logits + nvocab);
            top1.push_back(static_cast<int32_t>(best - logits));
        }
        // Write only when every layer and every required observation exists.
        write_capture(output_file, capture, top1, tokens);
        std::cout << "R10 capture complete: " << n
                  << " tokens, 24 layers, hidden=" << capture.hidden
                  << ". Private binary output only; no prompt content printed.\n";
    } catch (const std::exception & e) {
        std::cerr << "ZipBurger R10 rejected: " << e.what() << '\n';
        return 2;
    }
    return 0;
}
