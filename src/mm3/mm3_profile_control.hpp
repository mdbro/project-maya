#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace mm3 {
inline std::string profile_hex(const std::vector<uint32_t>& values) {
    static const char digits[]="0123456789abcdef";
    std::string out; out.reserve(values.size()*8);
    for (uint32_t v : values) for (int i=0;i<4;++i) {
        unsigned b=(v>>(8*i))&255; out.push_back(digits[b>>4]); out.push_back(digits[b&15]);
    }
    return out;
}
inline bool parse_profile_hex(const std::string& hex, std::vector<uint32_t>& values) {
    if (hex.empty() || hex.size()%8) return false;
    auto digit=[](char c)->int { if (c>='0'&&c<='9') return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return -1; };
    values.assign(hex.size()/8,0);
    for (size_t i=0;i<hex.size();i+=2) {
        int a=digit(hex[i]), b=digit(hex[i+1]); if(a<0||b<0) {values.clear();return false;}
        values[i/8] |= uint32_t((a<<4)|b) << (8*((i/2)%4));
    }
    return true;
}
inline bool generation_profile_accumulate(const std::vector<uint32_t>& before, const std::vector<uint32_t>& after,
                                          std::vector<uint32_t>& total) {
    if (before.size()!=after.size() || (!total.empty() && total.size()!=after.size())) return false;
    std::vector<uint32_t> next = total.empty() ? std::vector<uint32_t>(after.size(),0) : total;
    for (size_t i=0;i<after.size();++i) {
        const uint32_t delta=after[i]-before[i];
        if (UINT32_MAX-next[i]<delta) return false;
        next[i]+=delta;
    }
    total.swap(next); return true;
}
}
