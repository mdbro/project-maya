#pragma once
// Buffered diagnostic records. Never synchronize CUDA or flush per token here.
// A capture control file is read only between serialized model requests.
#include <nvtx3/nvToolsExt.h>
#include <zstd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace mm3::capture {
inline uint64_t clock_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline std::string quote(const std::string& s) {
    std::ostringstream o; o << '"';
    for (unsigned char c : s) {
        if(c=='"' || c=='\\') o << '\\' << char(c);
        else if(c=='\n') o << "\\n";
        else if(c=='\r') o << "\\r";
        else if(c=='\t') o << "\\t";
        else if(c<32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else o << char(c);
    }
    return o.str()+'"';
}
struct State {
    std::atomic<int> mode{0}; // 0 off, 1 token/round timeline, 2 detailed + NVTX
    std::atomic<uint64_t> epoch{0};
    std::atomic<long long> round{-1}, forward{-1}, layer{-1};
    std::mutex mutex;
    std::string experiment, run, request, directory;
    struct Row { std::string data; size_t original_size=0; };
    // Detailed rows are independently compressed in memory; output stays JSONL.
    std::vector<Row> rows;
    size_t bytes=0, dropped=0, detail_bytes=0, detail_dropped=0;
    uint64_t sequence=0;
    static constexpr size_t limit=512u*1024u*1024u;
    static constexpr size_t timeline_limit=64u*1024u*1024u;
    static constexpr size_t detail_limit=limit-timeline_limit;
};
inline State& state() { static State s; return s; }
inline bool enabled() { return state().mode.load(std::memory_order_relaxed)>0; }
inline bool detailed() { return state().mode.load(std::memory_order_relaxed)>1; }
inline void set_round(long long value) { if(enabled()) state().round.store(value,std::memory_order_relaxed); }
inline void set_forward(long long value) { if(enabled()) state().forward.store(value,std::memory_order_relaxed); }
inline void set_layer(long long value) { if(enabled()) state().layer.store(value,std::memory_order_relaxed); }
inline std::string context() {
    auto& s=state();
    return "experiment="+s.experiment+" run="+s.run+" request="+s.request+
        " round="+std::to_string(s.round.load())+" forward="+std::to_string(s.forward.load())+
        " layer="+std::to_string(s.layer.load());
}
// fields is an already escaped JSON member list without surrounding braces.
inline void record(const std::string& type, const std::string& fields, bool detail) {
    auto& s=state();const auto epoch=s.epoch.load(std::memory_order_acquire);
    if(!enabled() || (detail && !detailed())) return;
    std::lock_guard<std::mutex> guard(s.mutex);
    // A helper writer may have waited while end() closed the artifact.
    if(s.epoch.load(std::memory_order_relaxed)!=epoch || s.mode.load(std::memory_order_relaxed)==0 ||
       (detail && s.mode.load(std::memory_order_relaxed)<2))return;
    // Observation time under the record lock orders timestamps with event_seq,
    // including events emitted by helper threads.
    const auto stamp=clock_ns();
    const auto seq=++s.sequence;
    std::string row="{\"schema\":2,\"type\":"+quote(type)+",\"experiment_id\":"+quote(s.experiment)+
        ",\"run_id\":"+quote(s.run)+",\"request_id\":"+quote(s.request)+
        ",\"event_seq\":"+std::to_string(seq)+",\"steady_ns\":"+std::to_string(stamp)+
        ",\"round\":"+std::to_string(s.round.load())+",\"forward\":"+std::to_string(s.forward.load())+
        ",\"layer\":"+std::to_string(s.layer.load())+(fields.empty()?"":","+fields)+"}";
    // Compress each detail row independently, preserving its exact JSON bytes.
    // No file I/O takes place until end(); timeline remains uncompressed.
    State::Row stored;
    if(detail && row.size()>=128) {
        std::string packed(ZSTD_compressBound(row.size()), '\0');
        const size_t n=ZSTD_compress(&packed[0],packed.size(),row.data(),row.size(),1);
        if(!ZSTD_isError(n) && n<row.size()) {
            stored.data=std::string(packed.data(),n);
            stored.original_size=row.size();
        }
    }
    if(!stored.original_size) stored.data=std::move(row);
    // Include conservative vector growth and string allocation in the budget.
    const size_t charge=stored.data.capacity()+1+2*sizeof(State::Row);
    if(detail) {
        if(charge>State::detail_limit-s.detail_bytes) {++s.detail_dropped;return;}
        s.detail_bytes+=charge;
    } else {
        if(charge>State::timeline_limit-s.bytes) {++s.dropped;return;}
        s.bytes+=charge;
    }
    s.rows.push_back(std::move(stored));
}
inline void event(const std::string& type,const std::string& fields="") {record(type,fields,false);}
inline void detail_event(const std::string& type,const std::string& fields="") {record(type,fields,true);}
struct Range {
    bool active;
    explicit Range(const std::string& phase):active(detailed()) {
        if(active) nvtxRangePushA((context()+" phase="+phase).c_str());
    }
    ~Range() { if(active) nvtxRangePop(); }
    Range(const Range&)=delete; Range& operator=(const Range&)=delete;
};
inline bool safe_id(const std::string& s) {
    if(s.empty() || s.size()>160) return false;
    for(char c:s) if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'||c=='.'))return false;
    return s!="." && s!="..";
}
inline bool begin(std::string& error) {
    auto& s=state(); s.mode=0;s.epoch.fetch_add(1,std::memory_order_acq_rel);s.round=-1;s.forward=-1;s.layer=-1;
    {
        std::lock_guard<std::mutex> guard(s.mutex);
        s.rows.clear();s.bytes=0;s.dropped=0;s.detail_bytes=0;s.detail_dropped=0;s.sequence=0;
    }
    const char* control=std::getenv("MM3_CAPTURE_CONTROL");
    if(!control || !*control) return true;
    std::ifstream input(control); if(!input.good()) return true;
    std::string mode,extra;
    if(!(input>>mode>>s.experiment>>s.run>>s.request>>s.directory) || (input>>extra) ||
       !safe_id(s.experiment)||!safe_id(s.run)||!safe_id(s.request)||s.directory.empty()||s.directory[0]!='/') {
        error="invalid capture control file";return false;
    }
    if(mode=="off")return true;
    if(mode!="timeline" && mode!="detailed") {error="invalid capture mode";return false;}
    std::ifstream exists(s.directory+"/"+s.request+".jsonl");
    if(exists.good()) {error="capture request artifact already exists";return false;}
    const std::string marker=s.directory+"/"+s.request+".partial";
    std::ifstream prior(marker);
    if(prior.good()) {error="capture request partial marker already exists";return false;}
    std::ofstream pending(marker,std::ios::out);
    pending<<"{\"schema\":2,\"status\":\"partial\",\"experiment_id\":"<<quote(s.experiment)
           <<",\"run_id\":"<<quote(s.run)<<",\"request_id\":"<<quote(s.request)<<"}\n";
    pending.close();
    if(!pending){error="capture partial marker write failed";return false;}
    s.mode=mode=="detailed"?2:1;
    s.rows.reserve(4096);
    const auto wall=std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    event("capture_begin","\"mode\":"+quote(mode)+",\"wall_unix_ns\":"+std::to_string(wall)+
        ",\"clock\":\"std::chrono::steady_clock nanoseconds; NVTX clock alignment uses the shared marker\",\"buffer_limit_bytes\":"+std::to_string(State::limit)+
        ",\"timeline_limit_bytes\":"+std::to_string(State::timeline_limit)+",\"detail_limit_bytes\":"+std::to_string(State::detail_limit));
    if(detailed())nvtxMarkA((context()+" clock_sync_steady_ns="+std::to_string(clock_ns())).c_str());
    return true;
}
inline bool end(const std::string& status,std::string& error) {
    if(!enabled())return true;
    event("capture_end","\"status\":"+quote(status));
    auto& s=state();std::lock_guard<std::mutex> guard(s.mutex);
    std::ofstream output(s.directory+"/"+s.request+".jsonl",std::ios::out);
    for(const auto& row:s.rows) {
        if(row.original_size) {
            std::string decoded(row.original_size, '\0');
            const size_t n=ZSTD_decompress(&decoded[0],decoded.size(),row.data.data(),row.data.size());
            if(ZSTD_isError(n) || n!=row.original_size) {
                s.mode=0;error="capture buffer decompression failed";return false;
            }
            output<<decoded<<'\n';
        } else output<<row.data<<'\n';
    }
    output<<"{\"schema\":2,\"type\":\"capture_footer\",\"experiment_id\":"<<quote(s.experiment)
        <<",\"run_id\":"<<quote(s.run)<<",\"request_id\":"<<quote(s.request)
        <<",\"recorded_events\":"<<s.rows.size()<<",\"attempted_events\":"<<s.sequence
        <<",\"dropped_events\":"<<s.dropped+s.detail_dropped<<",\"timeline_dropped_events\":"<<s.dropped
        <<",\"detail_dropped_events\":"<<s.detail_dropped<<",\"buffered_bytes\":"<<s.bytes+s.detail_bytes
        <<",\"timeline_buffered_bytes\":"<<s.bytes<<",\"detail_buffered_bytes\":"<<s.detail_bytes
        <<",\"buffer_accounting\":\"compressed detail string capacity and conservative vector overhead\"}\n";
    output.close();s.mode=0;
    if(!output){error="capture artifact write failed";return false;}
    if(std::remove((s.directory+"/"+s.request+".partial").c_str())!=0){error="capture partial marker removal failed";return false;}
    return true;
}
}
