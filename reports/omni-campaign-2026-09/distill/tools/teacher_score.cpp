// teacher_score.cpp -- teacher-forced top-K scoring over libllama, for the Chimera Omni text teacher caches.
//
// Why: llama-server returns probabilities for GENERATED tokens only (it refuses "echo"), and a teacher cache needs
// the teacher's distribution at every stored position of a FIXED row.  This tool loads one GGUF once (any arch the
// linked libllama supports, e.g. qwen4exp = Qwen3.8-Flash-Next) and serves rows over a binary pipe:
//
//   stdout <- HELLO   u32 'TSH1', u32 json_bytes, json      (model + context facts, once, after the load)
//   stdin  -> REQUEST u32 'TSQ1', u32 n_tokens, u32 n_score, i32 tokens[n_tokens], u32 score_pos[n_score]
//             (score_pos strictly increasing, each < n_tokens; n_tokens == 0 = clean exit)
//   stdout <- REPLY   u32 'TSR1', u32 status (0 ok, 1 decode failed, 2 bad request), u32 n_score, u32 k,
//                     f64 decode_seconds, f64 post_seconds, f32 lse[n_score], i32 ids[n_score*k], f32 lp[n_score*k]
//
// Position t carries the teacher distribution over token t+1 given tokens [0, t] (the qtext_teacher_cache rule).
// Over logit rows [0, V) with V = min(n_vocab, --vocab-rows):  lse = logsumexp (accumulated in double), ids = the
// top-K rows descending by logit (ties: the lower id first), lp = logit[id] - lse.  The row is decoded in chunks of
// n_batch tokens on ONE sequence (the KV / recurrent state carries across chunks); the memory is cleared per row.
//
// Build (host g++, against the export lane's libllama build):  bash build_teacher_score.sh
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifndef TS_LLAMA_COMMIT
#define TS_LLAMA_COMMIT "unknown"
#endif

static const char * TOOL_VERSION = "1.0.1";
static const uint32_t MAGIC_HELLO = 0x31485354u;  // "TSH1"
static const uint32_t MAGIC_REQ = 0x31515354u;    // "TSQ1"
static const uint32_t MAGIC_REPLY = 0x31525354u;  // "TSR1"

struct Args {
    std::string model;
    int n_gpu_layers = 999;
    std::string split_mode = "layer";
    std::vector<float> tensor_split;
    int n_ctx = 8192;
    int n_batch = 1024;
    int n_ubatch = 512;
    int threads = 16;
    int post_threads = 16;
    int top_k = 64;
    int vocab_rows = 248320;
    std::string flash_attn = "auto";
    std::vector<std::string> cpu_patterns;  // --cpu-tensor PATTERN: keep matching tensors in host RAM
    bool verbose = false;
    bool no_mmap = false;   // read the GGUF once into memory (a network-mounted GGUF is then never touched again)
};

static void die(const char * msg) {
    fprintf(stderr, "teacher_score: %s\n", msg);
    exit(2);
}

static std::vector<float> parse_floats(const std::string & s) {
    std::vector<float> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) end = s.size();
        if (end > start) out.push_back(std::strtof(s.substr(start, end - start).c_str(), nullptr));
        start = end + 1;
    }
    return out;
}

static Args parse(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(("missing value for " + k).c_str());
            return argv[++i];
        };
        if (k == "-m" || k == "--model") a.model = next();
        else if (k == "-ngl" || k == "--n-gpu-layers") a.n_gpu_layers = std::atoi(next().c_str());
        else if (k == "-sm" || k == "--split-mode") a.split_mode = next();
        else if (k == "-ts" || k == "--tensor-split") a.tensor_split = parse_floats(next());
        else if (k == "-c" || k == "--ctx") a.n_ctx = std::atoi(next().c_str());
        else if (k == "-b" || k == "--batch") a.n_batch = std::atoi(next().c_str());
        else if (k == "-ub" || k == "--ubatch") a.n_ubatch = std::atoi(next().c_str());
        else if (k == "-t" || k == "--threads") a.threads = std::atoi(next().c_str());
        else if (k == "--post-threads") a.post_threads = std::atoi(next().c_str());
        else if (k == "--top-k") a.top_k = std::atoi(next().c_str());
        else if (k == "--vocab-rows") a.vocab_rows = std::atoi(next().c_str());
        else if (k == "-fa" || k == "--flash-attn") a.flash_attn = next();
        else if (k == "--cpu-tensor") a.cpu_patterns.push_back(next());
        else if (k == "-v" || k == "--verbose") a.verbose = true;
        else if (k == "--no-mmap") a.no_mmap = true;
        else die(("unknown argument " + k).c_str());
    }
    if (a.model.empty()) die("-m GGUF is required");
    if (a.top_k < 1 || a.top_k > 1024) die("--top-k must be in [1, 1024]");
    if (a.n_batch < 1 || a.n_ubatch < 1 || a.n_ubatch > a.n_batch) die("need 1 <= ubatch <= batch");
    if (a.post_threads < 1) a.post_threads = 1;
    return a;
}

static bool g_verbose = false;
static void log_cb(ggml_log_level level, const char * text, void *) {
    if (g_verbose || level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

static bool read_exact(void * dst, size_t n) {
    uint8_t * p = static_cast<uint8_t *>(dst);
    while (n) {
        size_t got = fread(p, 1, n, stdin);
        if (got == 0) return false;
        p += got;
        n -= got;
    }
    return true;
}

static void write_exact(const void * src, size_t n) {
    if (n && fwrite(src, 1, n, stdout) != n) die("short write on stdout");
}

template <typename T> static void put(const T & v) { write_exact(&v, sizeof(T)); }

static std::string json_escape(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

// One output row: lse over [0, V) and the top-K ids/logprobs, descending by logit, ties to the lower id.
static void score_row(const float * logits, int V, int K, float * lse_out, int32_t * ids_out, float * lp_out,
                      std::vector<std::pair<float, int32_t>> & heap) {
    float mx = -INFINITY;
    for (int j = 0; j < V; j++) mx = std::max(mx, logits[j]);
    double sum = 0.0;
    for (int j = 0; j < V; j++) sum += std::exp(static_cast<double>(logits[j]) - mx);
    const double lse = static_cast<double>(mx) + std::log(sum);
    // min-heap on (logit, -id): the root is the weakest kept entry; "stronger" = larger logit, then lower id
    auto weaker = [](const std::pair<float, int32_t> & x, const std::pair<float, int32_t> & y) {
        return x.first > y.first || (x.first == y.first && x.second < y.second);
    };
    heap.clear();
    for (int j = 0; j < V; j++) {
        const float v = logits[j];
        if (static_cast<int>(heap.size()) < K) {
            heap.emplace_back(v, j);
            std::push_heap(heap.begin(), heap.end(), weaker);
        } else if (v > heap.front().first) {
            std::pop_heap(heap.begin(), heap.end(), weaker);
            heap.back() = {v, j};
            std::push_heap(heap.begin(), heap.end(), weaker);
        }
    }
    std::sort(heap.begin(), heap.end(), [](const std::pair<float, int32_t> & x, const std::pair<float, int32_t> & y) {
        return x.first > y.first || (x.first == y.first && x.second < y.second);
    });
    *lse_out = static_cast<float>(lse);
    const int kept = static_cast<int>(heap.size());
    for (int i = 0; i < K; i++) {
        if (i < kept) {
            ids_out[i] = heap[i].second;
            lp_out[i] = static_cast<float>(static_cast<double>(heap[i].first) - lse);
        } else {  // V < K: cannot happen for a text teacher; refused upstream by the check on V
            ids_out[i] = -1;
            lp_out[i] = -INFINITY;
        }
    }
}

int main(int argc, char ** argv) {
    Args a = parse(argc, argv);
    g_verbose = a.verbose;
    llama_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = a.n_gpu_layers;
    if (a.no_mmap) { mp.load_mode = LLAMA_LOAD_MODE_NONE; mp.lazy_mode = LLAMA_LAZY_MODE_OFF; }  // lazy rows need mmap
    if (a.split_mode == "none") mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    else if (a.split_mode == "layer") mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
    else if (a.split_mode == "row") mp.split_mode = LLAMA_SPLIT_MODE_ROW;
    else die("--split-mode must be none, layer or row");
    std::vector<float> split(llama_max_devices(), 0.0f);
    if (!a.tensor_split.empty()) {
        if (a.tensor_split.size() > split.size()) die("--tensor-split names more devices than llama_max_devices()");
        std::copy(a.tensor_split.begin(), a.tensor_split.end(), split.begin());
        mp.tensor_split = split.data();
    }
    std::vector<llama_model_tensor_buft_override> overrides;
    if (!a.cpu_patterns.empty()) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) die("no CPU backend device for --cpu-tensor");
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_dev_buffer_type(cpu);
        for (const auto & p : a.cpu_patterns) overrides.push_back({p.c_str(), cpu_buft});
        overrides.push_back({nullptr, nullptr});
        mp.tensor_buft_overrides = overrides.data();
    }

    auto t_load = std::chrono::steady_clock::now();
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) die("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int V = std::min(n_vocab, a.vocab_rows);
    if (V < a.top_k) die("vocab rows < top_k");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(a.n_ctx);
    cp.n_batch = static_cast<uint32_t>(a.n_batch);
    cp.n_ubatch = static_cast<uint32_t>(a.n_ubatch);
    cp.n_seq_max = 1;
    cp.n_threads = a.threads;
    cp.n_threads_batch = a.threads;
    cp.no_perf = true;
    if (a.flash_attn == "on") cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    else if (a.flash_attn == "off") cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    else cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context init failed");
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();

    char desc[256] = {0};
    llama_model_desc(model, desc, sizeof desc);
    std::string hello = std::string("{\"tool\":\"teacher_score\",\"tool_version\":\"") + TOOL_VERSION +
        "\",\"llama_cpp_commit\":\"" + TS_LLAMA_COMMIT + "\",\"model\":\"" + json_escape(a.model) +
        "\",\"desc\":\"" + json_escape(desc) + "\",\"model_bytes\":" + std::to_string(llama_model_size(model)) +
        ",\"n_params\":" + std::to_string(llama_model_n_params(model)) +
        ",\"n_vocab\":" + std::to_string(n_vocab) + ",\"vocab_rows_scored\":" + std::to_string(V) +
        ",\"top_k\":" + std::to_string(a.top_k) + ",\"n_ctx\":" + std::to_string(llama_n_ctx(ctx)) +
        ",\"n_ctx_train\":" + std::to_string(llama_model_n_ctx_train(model)) +
        ",\"n_batch\":" + std::to_string(a.n_batch) + ",\"n_ubatch\":" + std::to_string(a.n_ubatch) +
        ",\"n_gpu_layers\":" + std::to_string(a.n_gpu_layers) + ",\"split_mode\":\"" + a.split_mode +
        "\",\"flash_attn\":\"" + a.flash_attn + "\",\"mmap\":" + (a.no_mmap ? "false" : "true") +
        ",\"cpu_tensor_patterns\":" + std::to_string(a.cpu_patterns.size()) +
        ",\"load_seconds\":" + std::to_string(load_s) + "}";
    put(MAGIC_HELLO);
    put(static_cast<uint32_t>(hello.size()));
    write_exact(hello.data(), hello.size());
    fflush(stdout);
    fprintf(stderr, "teacher_score %s ready: %s, n_vocab %d, V %d, ctx %u, load %.1f s\n", TOOL_VERSION, desc,
            n_vocab, V, llama_n_ctx(ctx), load_s);

    llama_batch batch = llama_batch_init(a.n_batch, 0, 1);
    std::vector<int32_t> tokens;
    std::vector<uint32_t> score_pos;
    std::vector<float> lse;
    std::vector<int32_t> ids;
    std::vector<float> lp;
    const int K = a.top_k;
    const int T = a.post_threads;
    std::vector<std::vector<std::pair<float, int32_t>>> heaps(T);
    for (auto & h : heaps) h.reserve(K + 1);

    while (true) {
        uint32_t magic = 0, n_tokens = 0, n_score = 0;
        if (!read_exact(&magic, 4)) break;
        if (magic != MAGIC_REQ) die("bad request magic");
        if (!read_exact(&n_tokens, 4) || !read_exact(&n_score, 4)) die("truncated request header");
        if (n_tokens == 0) break;
        tokens.resize(n_tokens);
        score_pos.resize(n_score);
        if (!read_exact(tokens.data(), 4ull * n_tokens) || !read_exact(score_pos.data(), 4ull * n_score))
            die("truncated request body");
        uint32_t status = 0;
        if (n_tokens > static_cast<uint32_t>(llama_n_ctx(ctx))) status = 2;
        for (uint32_t i = 0; i < n_score && status == 0; i++)
            if (score_pos[i] >= n_tokens || (i && score_pos[i] <= score_pos[i - 1])) status = 2;
        for (uint32_t i = 0; i < n_tokens && status == 0; i++)
            if (tokens[i] < 0 || tokens[i] >= n_vocab) status = 2;
        lse.assign(n_score, 0.0f);
        ids.assign(static_cast<size_t>(n_score) * K, -1);
        lp.assign(static_cast<size_t>(n_score) * K, -INFINITY);
        double decode_s = 0.0, post_s = 0.0;
        if (status == 0) {
            llama_memory_clear(llama_get_memory(ctx), true);
            uint32_t next_score = 0;  // index into score_pos
            for (uint32_t start = 0; start < n_tokens && status == 0; start += a.n_batch) {
                const uint32_t stop = std::min<uint32_t>(n_tokens, start + a.n_batch);
                batch.n_tokens = static_cast<int32_t>(stop - start);
                uint32_t s = next_score;
                std::vector<uint32_t> out_rows;  // score indices produced by this chunk, in output order
                for (uint32_t t = start; t < stop; t++) {
                    const int32_t j = static_cast<int32_t>(t - start);
                    batch.token[j] = tokens[t];
                    batch.pos[j] = static_cast<llama_pos>(t);
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = 0;
                    const bool want = s < n_score && score_pos[s] == t;
                    batch.logits[j] = want ? 1 : 0;
                    if (want) { out_rows.push_back(s); s++; }
                }
                auto t0 = std::chrono::steady_clock::now();
                const int rc = llama_decode(ctx, batch);
                llama_synchronize(ctx);
                auto t1 = std::chrono::steady_clock::now();
                decode_s += std::chrono::duration<double>(t1 - t0).count();
                if (rc != 0) { status = 1; fprintf(stderr, "teacher_score: llama_decode rc=%d\n", rc); break; }
                // the outputs of this chunk, scored in parallel across post threads
                const int n_out = static_cast<int>(out_rows.size());
                std::vector<const float *> rows_ptr(n_out);
                for (int o = 0; o < n_out; o++) {
                    const int32_t j = static_cast<int32_t>(score_pos[out_rows[o]] - start);
                    rows_ptr[o] = llama_get_logits_ith(ctx, j);
                    if (!rows_ptr[o]) { status = 1; break; }
                }
                if (status) break;
                std::vector<std::thread> pool;
                const int workers = std::min(T, std::max(1, n_out));
                for (int w = 0; w < workers; w++) {
                    pool.emplace_back([&, w]() {
                        for (int o = w; o < n_out; o += workers) {
                            const uint32_t si = out_rows[o];
                            score_row(rows_ptr[o], V, K, &lse[si], &ids[static_cast<size_t>(si) * K],
                                      &lp[static_cast<size_t>(si) * K], heaps[w]);
                        }
                    });
                }
                for (auto & th : pool) th.join();
                post_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
                next_score = s;
            }
            if (status == 0 && next_score != n_score) status = 2;
        }
        put(MAGIC_REPLY);
        put(status);
        put(n_score);
        put(static_cast<uint32_t>(K));
        put(decode_s);
        put(post_s);
        write_exact(lse.data(), 4ull * n_score);
        write_exact(ids.data(), 4ull * n_score * K);
        write_exact(lp.data(), 4ull * n_score * K);
        fflush(stdout);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
