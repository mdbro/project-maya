// CPU-only load/request head-default rules; no CUDA runtime or model.
// g++ -std=c++17 -Isrc/mm3 -Iinclude tests/mm3_head_request_rules_test.cpp -o /tmp/mm3_head_request_rules_test
#include "mm3_sample.hpp"
#include <cstdio>
#include <stdexcept>
using namespace mm3::sample;
static void need(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
static void fields(const Req& r, int mode, int format, int shape) {
    need(r.head_lowp==mode && r.head_lowp_fmt==format && r.head_shape==shape, "resolved head fields differ");
}
int main() {
    try {
        Req selected;
        need(selected.resolve_head_defaults(1,1), "selected startup");
        fields(selected,2,1,1);
        Req bf16;
        need(bf16.resolve_head_defaults(0,0), "omitted BF16 startup");
        fields(bf16,0,0,0);
        need(bf16.resolve_head_defaults(0,0), "BF16 resolution must be idempotent");
        fields(bf16,0,0,0);
        for (const char* key : {"head_lowp", "head_lowp_fmt", "head_shape"}) {
            Req explicit_zero;
            explicit_zero.head_lowp=explicit_zero.head_lowp_fmt=explicit_zero.head_shape=0;
            explicit_zero.given.insert(key);
            need(!explicit_zero.resolve_head_defaults(0,0), "explicit unsupported zero control was accepted");
            fields(explicit_zero,0,0,0);
            Req explicit_positive;
            explicit_positive.given.insert(key);
            need(!explicit_positive.resolve_head_defaults(0,0), "explicit unsupported positive control was accepted");
            fields(explicit_positive,2,1,1);
        }
        for (int formats : {2,3}) {
            Req nvfp4;
            need(nvfp4.resolve_head_defaults(formats,2), "NVFP4-first startup");
            fields(nvfp4,2,2,1);
            Req explicit_format;
            explicit_format.head_lowp_fmt=1;
            explicit_format.given.insert("head_lowp_fmt");
            need(explicit_format.resolve_head_defaults(formats,2), "explicit format resolution");
            fields(explicit_format,2,1,1); // The existing model setter validates whether this explicit format exists.
        }
        Req explicit_all;
        explicit_all.head_lowp=0; explicit_all.head_lowp_fmt=0; explicit_all.head_shape=0;
        explicit_all.given={"head_lowp","head_lowp_fmt","head_shape"};
        need(explicit_all.resolve_head_defaults(2,2), "explicit fields resolution");
        fields(explicit_all,0,0,0);
        // Reset uses a fresh Req with this same resolver: explicit fields from a prior request do not leak.
        Req reset;
        need(reset.resolve_head_defaults(2,2), "reset to startup first copy");
        fields(reset,2,2,1);
        need(head_parallel_start_supported(1,1,true), "selected single-request startup refused");
        for(int seqs : {2,4,8}) {
            need(head_parallel_start_supported(seqs,0,false), "explicit BF16-only parallel startup refused");
            need(!head_parallel_start_supported(seqs,0,true), "parallel missing-source drop accepted");
            for(int formats : {1,2,3})
                for(bool drop : {false,true})
                    need(!head_parallel_start_supported(seqs,formats,drop), "parallel lowp startup accepted");
        }
        std::puts("head request rules PASS: selected/BF16/NVFP4-first defaults, explicit keys, reset, parallel startup");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
