// tests/mm3_token_record_test.cpp - the routing lab's token records (src/mm3/mm3_token_record.hpp) and the routing
// path's hook (src/mm3/mm3_route_record.hpp) on the CPU:
//   1. the GEN keys (token_log, token_topk, force_tokens) and the records directory are refused when malformed;
//   2. the raw distribution is mm3_sample.hpp's own at temperature 1 with no filters, penalties included, id by id;
//      its top is ordered (p desc, id asc), top + rest is 1 and the entropy is the whole row's;
//   3. every line written is valid JSON with the fields API.md names; the writer creates exclusively and flushes;
//   4. SHA-256 against the standard's vectors; route_record's row mapping.
//     g++ -O2 -std=c++17 -Isrc/mm3 -Iinclude tests/mm3_token_record_test.cpp -o build/mm3_token_record_test
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>

#include "mm3_route_record.hpp"
#include "mm3_sample.hpp"
#include "mm3_token_record.hpp"

namespace tr = mm3::token_record;

static int failures = 0;
#define CHECK(c)                                                                \
    do {                                                                        \
        if (!(c)) {                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

// a minimal JSON validator (RFC 8259 grammar, no semantics): true when s is exactly one value
struct Json {
    const std::string& s;
    size_t i = 0;
    explicit Json(const std::string& v) : s(v) {}
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i; }
    bool lit(const char* w) { const size_t n = std::strlen(w); if (s.compare(i, n, w) != 0) return false; i += n; return true; }
    bool str() {
        if (s[i] != '"') return false;
        for (++i; i < s.size(); ++i) {
            const unsigned char c = (unsigned char) s[i];
            if (c == '"') { ++i; return true; }
            if (c < 32) return false;
            if (c == '\\') {
                if (++i >= s.size()) return false;
                if (s[i] == 'u') { for (int k = 0; k < 4; ++k) if (++i >= s.size() || !std::isxdigit((unsigned char) s[i])) return false; }
                else if (!std::strchr("\"\\/bfnrt", s[i])) return false;
            }
        }
        return false;
    }
    bool number() {
        const size_t a = i;
        if (s[i] == '-') ++i;
        if (i >= s.size() || !std::isdigit((unsigned char) s[i])) return false;
        if (s[i] == '0') ++i; else while (i < s.size() && std::isdigit((unsigned char) s[i])) ++i;
        if (i < s.size() && s[i] == '.') { ++i; if (i >= s.size() || !std::isdigit((unsigned char) s[i])) return false; while (i < s.size() && std::isdigit((unsigned char) s[i])) ++i; }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) { ++i; if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i; if (i >= s.size() || !std::isdigit((unsigned char) s[i])) return false; while (i < s.size() && std::isdigit((unsigned char) s[i])) ++i; }
        return i > a;
    }
    bool value() {
        ws();
        if (i >= s.size()) return false;
        bool ok;
        if (s[i] == '{') {
            ++i; ws();
            if (s[i] == '}') { ++i; ok = true; }
            else for (ok = false;;) { ws(); if (!str()) break; ws(); if (s[i] != ':') break; ++i; if (!value()) break; ws(); if (s[i] == ',') { ++i; continue; } if (s[i] == '}') { ++i; ok = true; } break; }
        } else if (s[i] == '[') {
            ++i; ws();
            if (s[i] == ']') { ++i; ok = true; }
            else for (ok = false;;) { if (!value()) break; ws(); if (s[i] == ',') { ++i; continue; } if (s[i] == ']') { ++i; ok = true; } break; }
        } else if (s[i] == '"') ok = str();
        else if (s[i] == 't') ok = lit("true");
        else if (s[i] == 'f') ok = lit("false");
        else if (s[i] == 'n') ok = lit("null");
        else ok = number();
        ws();
        return ok;
    }
    static bool valid(const std::string& v) { Json j(v); return j.value() && j.i == v.size(); }
};

static bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

static void keys() {
    tr::Settings s;
    bool bad = false;
    CHECK(tr::parse_key("token_log", "lab-3f2a9c_x.1", s, bad) && !bad && s.log == "lab-3f2a9c_x.1");
    for (const char* v : {"", ".", "..", "a/b", "../x", "a b", "a\\b"}) {
        tr::Settings t;
        CHECK(tr::parse_key("token_log", v, t, bad) && bad && t.log.empty());
    }
    CHECK(tr::parse_key("token_log", std::string(160, 'a'), s, bad) && !bad);
    CHECK(tr::parse_key("token_log", std::string(161, 'a'), s, bad) && bad);
    CHECK(tr::parse_key("token_topk", "1", s, bad) && !bad && s.topk == 1);
    CHECK(tr::parse_key("token_topk", "256", s, bad) && !bad && s.topk == 256);
    for (const char* v : {"0", "257", "-3", "", "6x", "1.5"}) CHECK(tr::parse_key("token_topk", v, s, bad) && bad);
    CHECK(s.topk == 256);   // (a refused value leaves the setting)
    CHECK(tr::parse_key("force_tokens", "5,0,199999", s, bad) && !bad && s.force == std::vector<int>({5, 0, 199999}));
    for (const char* v : {"", ",", "1,,2", "1,", ",1", "-1", "1,a", "1 2", "99999999999"}) {
        tr::Settings t;
        CHECK(tr::parse_key("force_tokens", v, t, bad) && bad && t.force.empty());
    }
    {
        std::string many;
        for (size_t k = 0; k < tr::kForceMax; ++k) many += (k ? ",7" : "7");
        tr::Settings t;
        CHECK(tr::parse_key("force_tokens", many, t, bad) && !bad && t.force.size() == tr::kForceMax);
        CHECK(tr::parse_key("force_tokens", many + ",7", t, bad) && bad);
    }
    CHECK(!tr::parse_key("routing_prior_bonus", "0.2", s, bad) && !bad);
    // the directory: absolute, no white space, an existing directory
    char tmpl[] = "/tmp/mm3_token_record_test_XXXXXX";
    const char* dir = mkdtemp(tmpl);
    CHECK(dir != nullptr);
    CHECK(tr::records_dir(nullptr).empty());
    CHECK(tr::records_dir("relative/dir").empty());
    CHECK(tr::records_dir("/no/such/dir/anywhere").empty());
    CHECK(tr::records_dir("/tmp/a b").empty());
    CHECK(tr::records_dir(dir) == dir);
    CHECK(tr::records_dir((std::string(dir) + "//").c_str()) == dir);
    rmdir(dir);
}

static void distributions() {
    std::mt19937_64 rng(11);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    const int V = 3000;
    int checked = 0;
    for (int it = 0; it < 300; ++it) {
        std::vector<float> row((size_t) V);
        for (float& x : row) x = nd(rng);
        for (int i = 0; i < 30; ++i) row[(size_t) (rng() % V)] = row[(size_t) (rng() % V)];   // (ties)
        if (it % 5 == 0) {
            const float top = *std::max_element(row.begin(), row.end());
            for (int i = 0; i < 12; ++i) row[(size_t) (rng() % V)] = top;                     // (ties at the top)
        }
        if (it % 7 == 0) for (int i = 0; i < 50; ++i) row[(size_t) (rng() % V)] = -INFINITY;  // (masked ids)
        mm3::sample::Req r;
        r.temperature = 1.0f;
        r.top_k = 0;
        r.top_p = 1.0f;
        r.min_p = 0.0f;
        std::vector<int> hist;
        for (int i = 0; i < 100; ++i) hist.push_back((int) (rng() % V));
        if (it % 2) {
            r.pen_n = it % 4 == 1 ? 16 : -1;
            r.pen_repeat = it % 3 ? 1.15f : 1.0f;
            r.pen_freq = 0.2f;
            r.pen_present = it % 3 == 2 ? -0.3f : 0.4f;
        }
        const auto whole = mm3::sample::dist(row.data(), V, r, hist);   // (temperature 1, no filters: every id)
        const tr::Penalty pen{r.pen_n, r.pen_repeat, r.pen_freq, r.pen_present};
        CHECK(pen.on() == mm3::sample::penalised(r));
        const int k = it % 3 == 0 ? 64 : it % 3 == 1 ? 1 : 256;
        tr::Row o;
        CHECK(tr::prepare(o, row.data(), V, pen, hist, whole, k, 3, 1, 2, ""));
        CHECK(o.valid && (int) o.top.size() == k);
        // every id's raw probability is the sampler's at temperature 1 (its sums run in another order: 1e-12)
        double maxrel = 0.0, h = 0.0;
        for (const auto& [id, p] : whole) {
            const double q = o.p_raw(id);
            if (p > 0) maxrel = std::max(maxrel, std::abs(q - p) / p);
            else CHECK(q == 0.0);
            if (p > 0) h -= p * std::log(p);
        }
        CHECK(maxrel < 1e-12);
        CHECK(std::abs(o.entropy - h) < 1e-9);
        // the top is the sampler's order (p desc, ties to the lower id) and top + rest is one
        double s = o.rest;
        for (int j = 0; j < k; ++j) {
            CHECK(o.top[(size_t) j].first == whole[(size_t) j].first);
            s += o.top[(size_t) j].second;
        }
        CHECK(std::abs(s - 1.0) < 1e-12);
        CHECK(o.p_samp(whole[0].first) == whole[0].second && o.p_samp(-5) == 0.0);
        ++checked;
    }
    CHECK(checked == 300);
    // a row the record cannot read: NaN or +inf
    std::vector<float> row(100, 0.5f);
    tr::Row o;
    row[7] = NAN;
    CHECK(!tr::prepare(o, row.data(), 100, {}, {}, {{1, 1.0}}, 8, 0, 0, 1, "") && !o.valid);
    row[7] = INFINITY;
    CHECK(!tr::prepare(o, row.data(), 100, {}, {}, {{1, 1.0}}, 8, 0, 0, 1, "") && !o.valid);
    CHECK(o.p_raw(1) == 0.0);
    const std::string line = tr::token_json(o, 1, 0, 10, false, false, nullptr);
    CHECK(Json::valid(line) && has(line, "\"invalid_row\":true") && has(line, "\"p\":1,"));
    // a uniform row: entropy log V, rest (V - k) / V
    std::vector<float> flat(1000, 2.0f);
    CHECK(tr::prepare(o, flat.data(), 1000, {}, {}, {{0, 1.0}}, 10, 0, 0, 1, ""));
    CHECK(std::abs(o.entropy - std::log(1000.0)) < 1e-12 && std::abs(o.rest - 0.99) < 1e-12);
    for (int j = 0; j < 10; ++j) CHECK(o.top[(size_t) j].first == j);   // (all tied: the lower ids)
}

static void lines() {
    std::vector<float> row(500);
    for (int i = 0; i < 500; ++i) row[(size_t) i] = (float) std::sin(i * 0.37) * 4.0f;
    const std::vector<std::pair<int, double>> d = {{40, 0.5}, {12, 0.3}, {7, 0.2}};
    tr::Row o;
    CHECK(tr::prepare(o, row.data(), 500, {}, {}, d, 2, 17, 3, 5, "{\"picks\":228,\"moved\":4}"));
    std::string t = tr::token_json(o, 12, 9, 1843, true, true, nullptr);
    CHECK(Json::valid(t));
    for (const char* f : {"\"type\":\"token\"", "\"i\":9", "\"pos\":1843", "\"id\":12", "\"p\":0.3,", "\"logp\":",
                          "\"p_raw\":", "\"logp_raw\":", "\"top\":[[", "\"rest\":", "\"entropy\":", "\"samp\":[[40,0.5],[12,0.3]]",
                          "\"samp_n\":3", "\"samp_rest\":0.2", "\"forced\":true", "\"source\":\"draft\"", "\"rows\":5",
                          "\"fwd\":17", "\"row\":3", "\"route\":{\"picks\":228,\"moved\":4}"})
        CHECK(has(t, f));
    CHECK(!has(t, "override"));
    t = tr::token_json(o, 99, 0, 1, false, false, "think_budget");   // (a token outside samp)
    CHECK(Json::valid(t) && has(t, "\"p\":0,\"logp\":null") && has(t, "\"override\":\"think_budget\"") &&
          has(t, "\"source\":\"target\"") && has(t, "\"route\":{\"picks\":228"));   // (the route is the row's)
    tr::Begin b;
    b.name = "lab-\"q\"";
    b.prompt_tokens = 1834;
    b.reused = 1790;
    b.forced = 512;
    b.max_new = 1024;
    b.sampling = {1.0f, 0.95f, 0.0f, 64, 7, false};
    b.spec = 1;
    b.draft_width = 7;
    b.vocab = 200064;
    const std::string bj = tr::begin_json(b);
    CHECK(Json::valid(bj) && has(bj, "\"type\":\"begin\"") && has(bj, "\"name\":\"lab-\\\"q\\\"\"") &&
          has(bj, "\"top_p\":0.949999988") && has(bj, "\"forced\":512") && has(bj, "\"vocab\":200064"));
    tr::Done dn;
    dn.tokens = 512;
    dn.finish = "forced";
    dn.begin = b;
    dn.commit = "abc123";
    dn.binary_sha256 = std::string(64, 'f');
    std::string dj = tr::done_json(dn);
    CHECK(Json::valid(dj) && has(dj, "\"finish\":\"forced\"") && !has(dj, "\"error\"") && !has(dj, "\"lab\"") &&
          !has(dj, "\"route\"") && has(dj, "\"binary_sha256\":\"ffff"));
    dn.finish = "error";
    dn.error = "line\nbreak";
    dn.lab = "{\"route_lab\":1}";
    dn.route = "{\"moved\":12}";
    dj = tr::done_json(dn);
    CHECK(Json::valid(dj) && has(dj, "\"error\":\"line\\nbreak\"") && has(dj, "\"lab\":{\"route_lab\":1}}") &&
          has(dj, "\"route\":{\"moved\":12}"));
    CHECK(tr::num(NAN) == "null" && tr::num(INFINITY) == "null" && tr::num(0.25) == "0.25");
    CHECK(!Json::valid("{\"a\":1,}") && !Json::valid("{\"a\":nan}") && !Json::valid("[1 2]"));   // (the checker checks)
    // the writer: exclusive, flushed a line at a time, a done line at the end
    char tmpl[] = "/tmp/mm3_token_record_test_XXXXXX";
    const char* dir = mkdtemp(tmpl);
    CHECK(dir != nullptr);
    std::string err;
    {
        tr::Writer w;
        CHECK(w.open(dir, "r1", err) && w.is_open());
        w.line(bj);
        std::ifstream peek(std::string(dir) + "/r1.jsonl");   // (flushed: readable while still open)
        std::string got;
        CHECK(std::getline(peek, got) && got == bj);
        tr::Writer w2;
        CHECK(!w2.open(dir, "r1", err) && !w2.is_open() && has(err, "cannot create"));
        w.finish(dj);
        CHECK(w.done() && !w.is_open());
        w.line("{\"late\":1}");   // (after done: nothing)
    }
    std::ifstream in(std::string(dir) + "/r1.jsonl");
    std::string l1, l2, l3;
    CHECK(std::getline(in, l1) && std::getline(in, l2) && !std::getline(in, l3) && l1 == bj && l2 == dj);
    std::remove((std::string(dir) + "/r1.jsonl").c_str());
    rmdir(dir);
}

static void digests() {
    auto hex = [](const std::string& s) { tr::Sha256 h; h.update(s.data(), s.size()); return h.hex(); };
    CHECK(hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(hex(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    {   // (in pieces of every size: the buffer's edges)
        const std::string m(1000, 'x');
        tr::Sha256 h;
        for (size_t a = 0, n = 1; a < m.size(); a += n, n = n % 67 + 1) h.update(m.data() + a, std::min(n, m.size() - a));
        CHECK(h.hex() == hex(m));
    }
    char path[] = "/tmp/mm3_token_record_sha_XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    CHECK(write(fd, "abc", 3) == 3);
    close(fd);
    CHECK(tr::sha256_file(path) == hex("abc"));
    std::remove(path);
    CHECK(tr::sha256_file("/no/such/file").empty());
}

static void hook() {
    namespace rr = mm3::route_record;
    const unsigned long long before = rr::forward().serial;
    CHECK(rr::row_text(before, 5, 5, 0).empty());   // (not filled this forward)
    rr::forward().serial++;
    rr::forward().rows = 5;
    rr::forward().row = {"{\"r\":0}", "{\"r\":1}", "{\"r\":2}", "{\"r\":3}", "{\"r\":4}"};
    CHECK(rr::row_text(before, 5, 5, 2) == "{\"r\":2}");
    CHECK(rr::row_text(before, 5, 1, 0) == "{\"r\":4}");   // (the prompt's last row: all n rows filled)
    CHECK(rr::row_text(before, 5, 5, 5).empty() && rr::row_text(before, 5, 5, -1).empty());
    rr::forward().row = {"{\"last\":1}"};                    // (only the logit rows filled)
    CHECK(rr::row_text(before, 5, 1, 0) == "{\"last\":1}");
    CHECK(rr::row_text(before, 5, 3, 0).empty());           // (neither n nor want rows: nothing)
    CHECK(rr::row_text(rr::forward().serial, 5, 1, 0).empty());
}

int main() {
    keys();
    distributions();
    lines();
    digests();
    hook();
    if (failures) {
        std::fprintf(stderr, "mm3_token_record_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("mm3_token_record_test: ok\n");
    return 0;
}
