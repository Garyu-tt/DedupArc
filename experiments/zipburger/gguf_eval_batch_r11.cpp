// R11: run many independent tiny GGUF-prefill probes with ONE model load.
// No generation, no writes to the model, CPU-only; 1..64 prompts, <=8 tokens
// each, sequentially, each with fresh KV state and a distinct output file.
// This is a developer research executable, NOT a model-level quality proof.
#define main zipburger_r10_single_entry_not_invoked
#include "gguf_eval_r10.cpp"
#undef main
#include <filesystem>

namespace fs = std::filesystem;

namespace {
std::vector<fs::path> prompt_files(const fs::path & directory) {
    if (!fs::is_directory(directory) || fs::is_symlink(directory)) {
        throw std::runtime_error("prompt directory must be an actual directory");
    }
    std::vector<fs::path> paths;
    for (const auto & entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() != ".txt") continue;
        if (entry.is_symlink() || !entry.is_regular_file()) {
            throw std::runtime_error("invalid non-regular prompt file");
        }
        paths.push_back(entry.path());
        if (paths.size() > 64) throw std::runtime_error("at most 64 prompts per run");
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) throw std::runtime_error("no .txt prompts found");
    return paths;
}
} // namespace

int main(int argc, char ** argv) {
#if !defined(_WIN32)
    umask(0077); // private output directory and all traced token IDs
#endif
    try {
        std::string model_file, in_dir, out_dir;
        int threads = 2;
        if (argc != 7 && argc != 9) {
            throw std::runtime_error(
                "usage: zipburger_gguf_batch_r11 --model /local/20b.gguf "
                "--prompt-dir /private/prompts --out-dir /private/new-traces "
                "[--threads 1..16]");
        }
        for (int i = 1; i < argc; i += 2) {
            const std::string key=argv[i], value=argv[i+1];
            if (key == "--model") model_file=value;
            else if (key == "--prompt-dir") in_dir=value;
            else if (key == "--out-dir") out_dir=value;
            else if (key == "--threads") threads=std::stoi(value);
            else throw std::runtime_error("unknown argument");
        }
        if (model_file.empty() || in_dir.empty() || out_dir.empty() ||
            threads < 1 || threads > 16) throw std::runtime_error("invalid bounded arguments");

        const auto paths = prompt_files(in_dir);
        if (!fs::create_directory(out_dir)) {
            throw std::runtime_error("output directory already exists; refusing overwrite");
        }
        llama_backend_init();
        struct Backend { ~Backend() { llama_backend_free(); } } backend_guard;
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_file.c_str(), params), llama_model_free);
        if (!model || llama_model_n_layer(model.get()) != kLayers ||
            llama_model_n_embd(model.get()) < 1 ||
            llama_model_n_embd(model.get()) > kMaxHidden) {
            throw std::runtime_error("expected local 24-layer GGUF model");
        }
        char architecture[80] = {};
        if (llama_model_meta_val_str(model.get(), "general.architecture",
                                     architecture, sizeof(architecture)) < 0 ||
            std::strcmp(architecture, "gpt-oss") != 0) {
            throw std::runtime_error("GGUF architecture must be gpt-oss");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        Capture capture;
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
        if (!ctx) throw std::runtime_error("cannot create private CPU inference context");
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
            llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        if (!batch) throw std::runtime_error("cannot allocate batch");

        size_t completed = 0;
        for (const fs::path & prompt_file : paths) {
            const std::string prompt = read_prompt(prompt_file.string());
            std::vector<llama_token> tokens(kMaxTokens);
            int count = llama_tokenize(vocab, prompt.c_str(), static_cast<int>(prompt.size()),
                                       tokens.data(), kMaxTokens, true, true);
            if (count < 1 || count > kMaxTokens) {
                throw std::runtime_error("one prompt exceeds the 8-token bound");
            }
            tokens.resize(static_cast<size_t>(count));

            // Clear the full inference state between independent examples.
            llama_memory_t memory = llama_get_memory(ctx.get());
            if (!memory) throw std::runtime_error("model context has no memory to reset");
            llama_memory_clear(memory, true);
            llama_batch_ext_clear(batch.get());
            capture = Capture{};
            capture.hidden = llama_model_n_embd(model.get());
            capture.tokens = count;
            for (int i = 0; i < count; ++i) {
                const int index = llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
                llama_pos pos = i;
                if (index < 0 || !llama_batch_ext_set_pos(batch.get(), index, &pos) ||
                    !llama_batch_ext_set_output_logits(batch.get(), index, true)) {
                    throw std::runtime_error("cannot add a prompt token");
                }
            }
            int rc = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (rc != 0 || capture.failed) {
                throw std::runtime_error("native callback/compute rejected prompt #" +
                                         std::to_string(completed));
            }
            const int vocab_size = llama_vocab_n_tokens(vocab);
            if (vocab_size < 1) throw std::runtime_error("invalid vocabulary");
            std::vector<int32_t> predictions;
            for (int i = 0; i < count; ++i) {
                const float * logits = llama_get_logits_ith(ctx.get(), i);
                if (!logits) throw std::runtime_error("missing per-token logits");
                predictions.push_back(static_cast<int32_t>(
                    std::max_element(logits, logits + vocab_size) - logits));
            }
            char output_name[32];
            std::snprintf(output_name, sizeof(output_name), "%03zu.zbr10", completed);
            write_capture((fs::path(out_dir) / output_name).string(),
                          capture, predictions, tokens);
            ++completed;
        }
        // This marker is written ONLY after all captures complete successfully.
        std::ofstream done((fs::path(out_dir)/"complete.marker").string(),
                           std::ios::binary | std::ios::out | std::ios::trunc);
        if (!done) throw std::runtime_error("failed to write completion marker");
        done << "ZBR11 COMPLETE " << completed << "\n";
        done.flush();
        if (!done) throw std::runtime_error("completion marker write failed");
        std::cout << "R11 batch complete: " << completed
                  << " bounded independent prompts, one GGUF model load.\n";
    } catch (const std::exception & e) {
        std::cerr << "ZipBurger R11 rejected: " << e.what() << '\n';
        return 2;
    }
    return 0;
}
