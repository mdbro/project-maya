// CPU-only capture fixture: g++ -std=c++17 -O2 -pthread -I/usr/local/cuda/include
// tests/mm3_capture_buffer_test.cpp -lzstd -o /tmp/mm3_capture_buffer_test
#include "../src/mm3/mm3_capture.hpp"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <unistd.h>

int main() {
    namespace c=mm3::capture;
    const auto dir=std::filesystem::temp_directory_path()/
        ("mm3-capture-buffer-"+std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    const auto control=dir/"control";
    auto start=[&](const char* id) {
        std::ofstream f(control);f<<"detailed experiment run "<<id<<' '<<dir.string()<<'\n';f.close();
        setenv("MM3_CAPTURE_CONTROL",control.c_str(),1);
        std::string error;assert(c::begin(error));
    };
    start("large");
    // Exceeds the previous 192 MiB detail budget before compression.
    std::string payload;payload.reserve(50000);
    for(int i=0;i<5000;++i)payload+="0.1234567,";
    const std::string fields="\"scores\":\""+payload+"\"";
    constexpr int count=4100;
    for(int i=0;i<count;++i)c::detail_event("route",fields);
    assert(c::state().detail_dropped==0);
    assert(c::state().detail_bytes < 16u*1024u*1024u);
    c::event("token_output","\"token\":42");
    std::string error;assert(c::end("ok",error));
    assert(!std::filesystem::exists(dir/"large.partial"));
    std::ifstream in(dir/"large.jsonl");std::string line;int routes=0;bool clean_footer=false;
    while(std::getline(in,line)) {
        if(line.find("\"type\":\"route\"")!=std::string::npos) {
            ++routes;assert(line.find(fields)!=std::string::npos);
        }
        clean_footer|=line.find("\"detail_dropped_events\":0")!=std::string::npos;
    }
    assert(routes==count && clean_footer);
    start("overflow");
    c::state().detail_bytes=c::State::detail_limit;
    c::detail_event("must_drop",fields);
    c::event("timeline_survives");
    assert(c::end("ok",error));
    std::ifstream overflow(dir/"overflow.jsonl");bool timeline=false,footer=false;
    while(std::getline(overflow,line)) {
        timeline|=line.find("timeline_survives")!=std::string::npos;
        footer|=line.find("\"detail_dropped_events\":1")!=std::string::npos;
    }
    assert(timeline && footer);
    std::filesystem::remove_all(dir);
    std::cout<<"capture buffer round trip and overflow passed\n";
}
