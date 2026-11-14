// src/mm3/mm3_token_record.hpp - the routing lab's per-token records (routing-lab-20261110/API.md, section 1): a request
// that names `token_log` gets one JSON line per committed token in <MM3_TOKEN_RECORDS_DIR>/<token_log>.jsonl, flushed as
// the token goes out, between a `begin` and a `done` line.  Each token line carries two distributions of the row that
// produced it: raw (softmax of the logits at temperature 1 after the request's penalties, before temperature, top_k,
// top_p and min_p: the server's logprobs convention) and samp (the sampler's: what the token was drawn from or a draft
// was judged against).  force_tokens: the ids to commit in order (forced decoding; mm3_main.cpp acts on them).
// Header-only and CUDA-free, so tests/mm3_token_record_test.cpp checks it on the CPU.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mm3::token_record {

constexpr int kTopkDefault = 64, kTopkMax = 256;
constexpr size_t kForceMax = 65536;

/// A request's keys (the GEN line's token_log=, token_topk=, force_tokens=).
struct Settings {
    std::string log;         // the file's name; empty: no records
    int topk = kTopkDefault; // entries in "top" and in "samp"
    std::vector<int> force;  // forced decoding's ids; empty: none
};

inline bool safe_name(const std::string& s) {   // (as capture::safe_id: no path, no dot names)
    if (s.empty() || s.size() > 160 || s == "." || s == "..") return false;
    for (char c : s)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' ||
              c == '.'))
            return false;
    return true;
}

/// One GEN key: true when it is one of these (then `bad` says whether its value was refused).
inline bool parse_key(const std::string& k, const std::string& v, Settings& s, bool& bad) {
    bad = false;
    if (k == "token_log") {
        bad = !safe_name(v);
        if (!bad) s.log = v;
        return true;
    }
    if (k == "token_topk") {
        char* end = nullptr;
        const long n = std::strtol(v.c_str(), &end, 10);
        bad = end == v.c_str() || *end || n < 1 || n > kTopkMax;
        if (!bad) s.topk = (int) n;
        return true;
    }
    if (k == "force_tokens") {
        std::vector<int> ids;
        size_t a = 0;
        while (!bad && a <= v.size()) {
            const size_t b = std::min(v.find(',', a), v.size());
            const std::string f = v.substr(a, b - a);
            char* end = nullptr;
            const long n = std::strtol(f.c_str(), &end, 10);
            if (f.empty() || end == f.c_str() || *end || n < 0 || n > 0x7fffffffL || ids.size() >= kForceMax) bad = true;
            else ids.push_back((int) n);
            a = b + 1;
        }
        bad = bad || ids.empty();
        if (!bad) s.force = std::move(ids);
        return true;
    }
    return false;
}

/// The directory of MM3_TOKEN_RECORDS_DIR when it is usable (absolute, no white space - it goes on the INFO line - and
/// an existing directory), else "".
inline std::string records_dir(const char* v) {
#ifdef _WIN32
    (void) v;
    return "";   // (the lab runs on Linux; records stay off elsewhere)
#else
    if (!v || v[0] != '/') return "";
    std::string d(v);
    for (char c : d)
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return "";
    while (d.size() > 1 && d.back() == '/') d.pop_back();
    struct stat st {};
    if (stat(d.c_str(), &st) != 0 || (st.st_mode & S_IFMT) != S_IFDIR) return "";
    return d;
#endif
}

/// The request's penalties as mm3_sample.hpp's dist_c applies them (llama.cpp's, over the history's last n tokens).
struct Penalty {
    int n = 0;
    float repeat = 1.0f, freq = 0.0f, present = 0.0f;
    bool on() const { return n != 0 && (repeat != 1.0f || freq != 0.0f || present != 0.0f); }
};

inline void apply_penalty(std::vector<float>& y, const Penalty& pen, const std::vector<int>& hist) {
    if (!pen.on()) return;
    std::unordered_map<int, int> cnt;
    const size_t n = pen.n < 0 ? hist.size() : std::min(hist.size(), (size_t) pen.n);
    for (size_t i = hist.size() - n; i < hist.size(); ++i) ++cnt[hist[i]];
    for (const auto& [t, k] : cnt) {
        if (t < 0 || t >= (int) y.size()) continue;
        float& x = y[(size_t) t];   // (dist_c's own float steps, in its order)
        if (pen.repeat > 0.0f && pen.repeat != 1.0f) x = x > 0 ? x / pen.repeat : x * pen.repeat;
        x -= pen.freq * (float) k + pen.present;
    }
}

/// One row's distributions, prepared where its token is drawn and written when the token goes out.
struct Row {
    std::vector<float> y;                       // the row's logits after penalties: raw = softmax(y)
    std::vector<double> w;                      // exp(y - m), 0 for -inf
    float m = 0.0f;                             // max of y
    double z = 0.0;                             // sum of w
    std::vector<std::pair<int, double>> top;    // raw, the topk largest (p desc, id asc)
    double rest = 0.0, entropy = 0.0;           // raw mass outside top; raw entropy (nats, whole vocabulary)
    std::vector<std::pair<int, double>> samp;   // the sampler's distribution, whole (most likely first)
    int topk = kTopkDefault;
    long long fwd = -1;                         // the forward's index in the request (0: the prompt's last call)
    int row = 0, rows = 0;                      // the row in it, and its rows
    std::string route;                          // route_record's text for the row, "" none
    bool valid = false;

    double p_raw(int t) const {
        if (!valid || t < 0 || t >= (int) w.size()) return 0.0;
        return w[(size_t) t] / z;
    }
    double p_samp(int t) const {
        for (const auto& [id, q] : samp)
            if (id == t) return q;
        return 0.0;
    }
};

/// Row o from logits x[V] (as the model returned them), the history the sampler penalised over, and the sampler's
/// distribution d.  false (o.valid false) for a row with NaN or +inf.
inline bool prepare(Row& o, const float* x, int V, const Penalty& pen, const std::vector<int>& hist,
                    const std::vector<std::pair<int, double>>& d, int topk, long long fwd, int row, int rows,
                    const std::string& route) {
    o.valid = false;
    o.fwd = fwd;
    o.row = row;
    o.rows = rows;
    o.route = route;
    o.topk = std::max(1, std::min(topk, kTopkMax));
    o.samp = d;
    o.top.clear();
    o.rest = o.entropy = 0.0;
    if (V < 1 || !x) return false;
    o.y.assign(x, x + V);
    apply_penalty(o.y, pen, hist);
    float m = -INFINITY;
    for (float v : o.y) {
        if (std::isnan(v) || v == INFINITY) return false;
        m = std::max(m, v);
    }
    if (!std::isfinite(m)) return false;
    o.m = m;
    o.w.resize((size_t) V);
    double z = 0.0, s = 0.0;
    for (size_t i = 0; i < (size_t) V; ++i) {
        const float v = o.y[i];
        if (!std::isfinite(v)) { o.w[i] = 0.0; continue; }
        const double dv = (double) (v - m);   // (dist_c's step: a float difference, a double exponential)
        const double w = std::exp(dv);
        o.w[i] = w;
        z += w;
        s += w * dv;
    }
    o.z = z;
    o.entropy = std::log(z) - s / z;
    const int k = std::min(o.topk, V);
    std::vector<int> idx((size_t) V);
    std::iota(idx.begin(), idx.end(), 0);
    const auto before = [&](int a, int b) {
        return o.y[(size_t) a] > o.y[(size_t) b] || (o.y[(size_t) a] == o.y[(size_t) b] && a < b);
    };
    std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(), before);
    std::sort(idx.begin(), idx.begin() + k, before);
    o.top.reserve((size_t) k);
    o.valid = true;
    for (int j = 0; j < k; ++j) o.top.push_back({idx[(size_t) j], o.p_raw(idx[(size_t) j])});
    double rest = 0.0;   // (summed, not 1 - top: a small rest keeps its digits)
    for (size_t j = (size_t) k; j < idx.size(); ++j) rest += o.w[(size_t) idx[j]];
    o.rest = rest / z;
    return true;
}

/// A JSON number with 9 significant digits ("null" for a value JSON cannot hold).
inline std::string num(double v) {
    if (!std::isfinite(v)) return "null";
    char b[40];
    std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}

inline std::string pairs(const std::vector<std::pair<int, double>>& v, size_t k) {
    std::string s = "[";
    for (size_t j = 0; j < v.size() && j < k; ++j) {
        if (j) s += ',';
        s += '[' + std::to_string(v[j].first) + ',' + num(v[j].second) + ']';
    }
    return s + ']';
}

/// The token line for token id, committed from row o as output index i at context position pos.
// lookup: a kept draft of a prompt lookup round (a copied token, verified with q = 1): "lookup":true after "source"
// (absent otherwise, so a record without the lookup is as before)
inline std::string token_json(const Row& o, int id, long long i, long long pos, bool forced, bool draft,
                              const char* override_reason, bool lookup = false) {
    const double p = o.p_samp(id), pr = o.p_raw(id);
    std::string s = "{\"type\":\"token\",\"i\":" + std::to_string(i) + ",\"pos\":" + std::to_string(pos) +
                    ",\"id\":" + std::to_string(id);
    s += ",\"p\":" + num(p) + ",\"logp\":" + (p > 0 ? num(std::log(p)) : std::string("null"));
    s += ",\"p_raw\":" + num(pr) + ",\"logp_raw\":" + (pr > 0 ? num(std::log(pr)) : std::string("null"));
    if (o.valid) {
        s += ",\"top\":" + pairs(o.top, o.top.size()) + ",\"rest\":" + num(o.rest) + ",\"entropy\":" + num(o.entropy);
    } else {
        s += ",\"top\":[],\"rest\":null,\"entropy\":null,\"invalid_row\":true";
    }
    double samp_rest = 0.0;
    for (size_t j = (size_t) o.topk; j < o.samp.size(); ++j) samp_rest += o.samp[j].second;
    s += ",\"samp\":" + pairs(o.samp, (size_t) o.topk) + ",\"samp_n\":" + std::to_string(o.samp.size()) +
         ",\"samp_rest\":" + num(samp_rest);
    s += std::string(",\"forced\":") + (forced ? "true" : "false") + ",\"source\":" + (draft ? "\"draft\"" : "\"target\"");
    if (draft && lookup) s += ",\"lookup\":true";
    s += ",\"rows\":" + std::to_string(o.rows) + ",\"fwd\":" + std::to_string(o.fwd) + ",\"row\":" + std::to_string(o.row);
    if (override_reason && *override_reason) s += std::string(",\"override\":\"") + override_reason + '"';
    if (!o.route.empty()) s += ",\"route\":" + o.route;
    return s + '}';
}

inline std::string quote(const std::string& v) {
    std::string s = "\"";
    for (unsigned char c : v) {
        if (c == '"' || c == '\\') { s += '\\'; s += (char) c; }
        else if (c == '\n') s += "\\n";
        else if (c == '\r') s += "\\r";
        else if (c == '\t') s += "\\t";
        else if (c < 32) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); s += b; }
        else s += (char) c;
    }
    return s + '"';
}

struct Sampling {
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 0;
    unsigned long long seed = 0;
    bool penalties = false;
};

inline std::string sampling_json(const Sampling& g) {
    return "{\"temperature\":" + num(g.temperature) + ",\"top_p\":" + num(g.top_p) + ",\"top_k\":" +
           std::to_string(g.top_k) + ",\"min_p\":" + num(g.min_p) + ",\"seed\":" + std::to_string(g.seed) +
           ",\"penalties\":" + (g.penalties ? "true" : "false") + '}';
}

struct Begin {
    std::string name;
    size_t prompt_tokens = 0, reused = 0, forced = 0;
    long long max_new = 0;
    Sampling sampling;
    int spec = 0, draft_width = 0, topk = kTopkDefault, vocab = 0;
};

inline std::string begin_json(const Begin& b) {
    return "{\"type\":\"begin\",\"schema\":1,\"name\":" + quote(b.name) + ",\"prompt_tokens\":" +
           std::to_string(b.prompt_tokens) + ",\"reused\":" + std::to_string(b.reused) + ",\"max_new\":" +
           std::to_string(b.max_new) + ",\"forced\":" + std::to_string(b.forced) + ",\"sampling\":" +
           sampling_json(b.sampling) + ",\"spec\":" + std::to_string(b.spec) + ",\"draft_width\":" +
           std::to_string(b.draft_width) + ",\"token_topk\":" + std::to_string(b.topk) + ",\"vocab\":" +
           std::to_string(b.vocab) + '}';
}

struct Done {
    long long tokens = 0, forwards = 0, drafts_offered = 0, drafts_kept = 0;
    std::string finish = "error", error;
    double elapsed_ms = 0.0;
    Begin begin;              // (its settings again)
    std::string lab, route;   // route_record::request()'s settings and summary, "" none
    std::string commit, binary_sha256;
};

inline std::string done_json(const Done& d) {
    std::string s = "{\"type\":\"done\",\"tokens\":" + std::to_string(d.tokens) + ",\"finish\":" + quote(d.finish);
    if (!d.error.empty()) s += ",\"error\":" + quote(d.error);
    s += ",\"forwards\":" + std::to_string(d.forwards) + ",\"drafts_offered\":" + std::to_string(d.drafts_offered) +
         ",\"drafts_kept\":" + std::to_string(d.drafts_kept) + ",\"elapsed_ms\":" + num(d.elapsed_ms);
    s += ",\"settings\":{\"token_topk\":" + std::to_string(d.begin.topk) + ",\"forced\":" +
         std::to_string(d.begin.forced) + ",\"sampling\":" + sampling_json(d.begin.sampling) + ",\"spec\":" +
         std::to_string(d.begin.spec) + ",\"draft_width\":" + std::to_string(d.begin.draft_width);
    if (!d.lab.empty()) s += ",\"lab\":" + d.lab;
    s += '}';
    if (!d.route.empty()) s += ",\"route\":" + d.route;
    s += ",\"engine\":{\"commit\":" + quote(d.commit) + ",\"binary_sha256\":" + quote(d.binary_sha256) + "}}";
    return s;
}

/// The request's file: created exclusively, a line flushed at a time.  A failed write stops the file (the request
/// itself goes on) and says so once on stderr.
class Writer {
public:
    Writer() = default;
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    ~Writer() { close(); }
    bool open(const std::string& dir, const std::string& name, std::string& err) {
        close();
        path_ = dir + "/" + name + ".jsonl";
        f_ = std::fopen(path_.c_str(), "wx");   // (C11 exclusive create: an existing file is refused)
        if (!f_) {
            err = "token_log: cannot create " + path_ + " (it exists, or the directory is not writable)";
            return false;
        }
        done_ = failed_ = false;
        return true;
    }
    bool is_open() const { return f_ != nullptr; }
    bool done() const { return done_; }
    void line(const std::string& s) {
        if (!f_ || failed_) return;
        if (std::fwrite(s.data(), 1, s.size(), f_) != s.size() || std::fputc('\n', f_) == EOF || std::fflush(f_) != 0) {
            failed_ = true;
            std::fprintf(stderr, "maya-mm3: token records: writing %s failed; the rest of this request has none\n",
                         path_.c_str());
        }
    }
    void finish(const std::string& done_line) {
        if (!f_ || done_) return;
        line(done_line);
        done_ = true;
        close();
    }
    void close() {
        if (f_) std::fclose(f_);
        f_ = nullptr;
    }
    const std::string& path() const { return path_; }

private:
    std::FILE* f_ = nullptr;
    std::string path_;
    bool done_ = false, failed_ = false;
};

/// SHA-256 (FIPS 180-4), for the engine binary's digest in the done line.
class Sha256 {
public:
    Sha256() { std::memcpy(h_, kInit, sizeof h_); }
    void update(const void* data, size_t n) {
        const unsigned char* p = static_cast<const unsigned char*>(data);
        len_ += n;
        while (n > 0) {
            const size_t take = std::min(n, (size_t) 64 - used_);
            std::memcpy(buf_ + used_, p, take);
            used_ += take;
            p += take;
            n -= take;
            if (used_ == 64) {
                block(buf_);
                used_ = 0;
            }
        }
    }
    std::string hex() {
        const uint64_t bits = len_ * 8;
        const unsigned char one = 0x80, zero = 0;
        update(&one, 1);
        while (used_ != 56) update(&zero, 1);
        unsigned char be[8];
        for (int i = 0; i < 8; ++i) be[i] = (unsigned char) (bits >> (56 - 8 * i));
        update(be, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h_[i]);
        return std::string(out, 64);
    }

private:
    static constexpr uint32_t kInit[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    static uint32_t rotr(uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }
    void block(const unsigned char* p) {
        static const uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t) p[4 * i] << 24 | (uint32_t) p[4 * i + 1] << 16 | (uint32_t) p[4 * i + 2] << 8 |
                   (uint32_t) p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
    }
    uint32_t h_[8];
    unsigned char buf_[64] = {};
    size_t used_ = 0;
    uint64_t len_ = 0;
};

/// The SHA-256 of a file, "" when it cannot be read.
inline std::string sha256_file(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return "";
    Sha256 h;
    std::vector<unsigned char> buf(1 << 20);
    size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) h.update(buf.data(), n);
    const bool ok = !std::ferror(f);
    std::fclose(f);
    return ok ? h.hex() : "";
}

}  // namespace mm3::token_record
