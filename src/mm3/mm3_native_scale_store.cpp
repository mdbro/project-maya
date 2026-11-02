#include "mm3_native_scale_store.hpp"
#include <zstd.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace mm3::scale_store {
namespace {
constexpr uint32_t kMagic = 0x3153434d; // MCS1, little-endian
constexpr uint32_t kVersion = 1;
constexpr size_t kUnits = kScaleBytes / 16;
constexpr size_t kData = kHeaderBytes + kDescriptorBytes + kCodeBytes;
size_t align256(size_t n) { return (n + 255) & ~size_t(255); }
void put32(uint8_t* p, uint32_t x) { for (int i=0;i<4;++i) p[i]=uint8_t(x>>(8*i)); }
uint32_t get32(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1])<<8 | uint32_t(p[2])<<16 | uint32_t(p[3])<<24; }
bool fail(std::string& e, const char* s) { e=s; return false; }
// Emits the exact RG16 scale plane without touching a code tensor.
void scale_plane(const uint8_t* g, const uint8_t* u, const uint8_t* d, uint8_t* out) {
    size_t pos=0;
    for(size_t group=0;group<384;++group) for(size_t col=0;col<384;++col)
        for(size_t row=0;row<16;++row) {
            size_t r=group*16+row;
            out[pos++]=(r&1 ? u : g)[(r/2)*384+col];
        }
    for(size_t group=0;group<384;++group) for(size_t col=0;col<192;++col)
        for(size_t row=0;row<16;++row) out[pos++]=d[(group*16+row)*192+col];
}
bool compress_plane(const uint8_t* plane, uint8_t* dst, size_t capacity,
                    uint32_t offsets[kBlocks+1], size_t& bytes, std::string& err) {
    std::vector<uint8_t> tmp(ZSTD_compressBound(kBlockBytes));
    bytes=0; offsets[0]=0;
    for(size_t b=0;b<kBlocks;++b) {
        size_t n=ZSTD_compress(tmp.data(),tmp.size(),plane+b*kBlockBytes,kBlockBytes,1);
        if(ZSTD_isError(n)) { err=ZSTD_getErrorName(n); return false; }
        if(n>std::numeric_limits<uint32_t>::max()-bytes) return fail(err,"compressed scale size overflow");
        if(dst) { if(bytes>capacity || n>capacity-bytes) return fail(err,"encoded record does not fit slot"); std::memcpy(dst+bytes,tmp.data(),n); }
        bytes+=n; offsets[b+1]=uint32_t(bytes);
    }
    return true;
}
}
bool encoded_size_from_scales(const uint8_t* gate,const uint8_t* up,const uint8_t* down,size_t& result,std::string& err) {
    if(!gate||!up||!down) return fail(err,"null official scale tensor");
    std::vector<uint8_t> plane(kScaleBytes); scale_plane(gate,up,down,plane.data());
    uint32_t offsets[kBlocks+1];size_t n;
    if(!compress_plane(plane.data(),nullptr,0,offsets,n,err)) return false;
    result=align256(kData+n); return true;
}
bool encode(const uint8_t* raw,size_t raw_bytes,uint8_t* dst,size_t cap,size_t& result,std::string& err) {
    if(!raw||!dst||raw_bytes!=kRawBytes||cap<kData) return fail(err,"invalid raw record or encoded capacity");
    std::vector<uint8_t> plane(kScaleBytes);
    std::memcpy(dst,raw,kHeaderBytes);
    for(size_t i=0;i<kUnits;++i) {
        const uint8_t* unit=raw+kHeaderBytes+i*144;
        std::memcpy(dst+kHeaderBytes+kDescriptorBytes+i*128,unit,128);
        std::memcpy(plane.data()+i*16,unit+128,16);
    }
    uint32_t offsets[kBlocks+1];size_t n;
    if(!compress_plane(plane.data(),dst+kData,cap-kData,offsets,n,err)) return false;
    result=align256(kData+n); if(result>cap) return fail(err,"encoded record alignment exceeds slot");
    auto* desc=dst+kHeaderBytes; std::memset(desc,0,kDescriptorBytes);
    put32(desc,kMagic);put32(desc+4,kVersion);put32(desc+8,uint32_t(result));
    for(size_t i=0;i<=kBlocks;++i) put32(desc+12+i*4,offsets[i]);
    std::memset(dst+kData+n,0,result-(kData+n)); return true;
}
bool view(const uint8_t* src,size_t cap,View& out,std::string& err) {
    if(!src||cap<kData) return fail(err,"truncated encoded record");
    const auto* desc=src+kHeaderBytes;
    if(get32(desc)!=kMagic||get32(desc+4)!=kVersion) return fail(err,"invalid scale codec magic or version");
    const size_t total=get32(desc+8);
    if(total<kData || total>cap || total%256) return fail(err,"invalid encoded record size");
    View v; v.header=src;v.codes=src+kHeaderBytes+kDescriptorBytes;v.compressed=src+kData;
    for(size_t i=0;i<=kBlocks;++i) v.offsets[i]=get32(desc+12+i*4);
    if(v.offsets[0]!=0) return fail(err,"invalid first scale offset");
    for(size_t i=0;i<kBlocks;++i) if(v.offsets[i]>=v.offsets[i+1]) return fail(err,"invalid scale block offsets");
    size_t n=v.offsets[kBlocks];
    if(n>total-kData || align256(kData+n)!=total) return fail(err,"scale data exceeds encoded record");
    for(size_t i=232;i<kDescriptorBytes;++i) if(desc[i]) return fail(err,"nonzero reserved codec metadata");
    out=v;return true;
}
bool validate(const uint8_t* src,size_t cap,std::string& err) { View v;return view(src,cap,v,err); }
bool decode_block(const View& v,size_t b,uint8_t* scales,std::string& err) {
    if(b>=kBlocks||!scales||!v.compressed) return fail(err,"invalid scale block request");
    size_t n=ZSTD_decompress(scales+b*kBlockBytes,kBlockBytes,v.compressed+v.offsets[b],v.offsets[b+1]-v.offsets[b]);
    if(ZSTD_isError(n)) {err=ZSTD_getErrorName(n);return false;}
    if(n!=kBlockBytes) return fail(err,"incorrect decoded scale block size");return true;
}
bool decode_raw(const uint8_t* src,size_t cap,uint8_t* raw,size_t raw_cap,std::string& err) {
    if(!raw||raw_cap<kRawBytes) return fail(err,"insufficient raw record capacity");
    View v;if(!view(src,cap,v,err)) return false;
    std::vector<uint8_t> plane(kScaleBytes);
    for(size_t b=0;b<kBlocks;++b) if(!decode_block(v,b,plane.data(),err)) return false;
    std::memcpy(raw,v.header,kHeaderBytes);
    for(size_t i=0;i<kUnits;++i) {auto* unit=raw+kHeaderBytes+i*144;std::memcpy(unit,v.codes+i*128,128);std::memcpy(unit+128,plane.data()+i*16,16);}
    return true;
}
}
