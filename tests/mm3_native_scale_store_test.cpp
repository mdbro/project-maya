// Standalone CPU codec test. Pass the directory containing the six retained
// l{3,30,59}-e{0,1}.rg16 records. Links the production codec with libzstd.
#include "mm3/mm3_native_scale_store.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace sc=mm3::scale_store;
static void require(bool ok,const std::string& what) {if(!ok)throw std::runtime_error(what);}
static void set32(uint8_t* p,uint32_t x){for(int i=0;i<4;++i)p[i]=uint8_t(x>>(i*8));}
int main(int argc,char** argv) try {
    require(argc==2,"expected six-record directory");
    std::vector<uint8_t> raw(sc::kRawBytes),back(sc::kRawBytes);
    std::vector<uint8_t> gate(3072*384),up(gate.size()),down(6144*192);
    // Worst case bound is independent of inventory assumptions.
    std::vector<uint8_t> enc(sc::kRawBytes+1024*1024);
    for(int layer:{3,30,59})for(int expert:{0,1}) {
        std::string path=std::string(argv[1])+"/l"+std::to_string(layer)+"-e"+std::to_string(expert)+".rg16";
        std::ifstream in(path,std::ios::binary|std::ios::ate);
        require(bool(in)&&in.tellg()==std::streamoff(sc::kRawBytes-sc::kHeaderBytes+12),"wrong fixture size: "+path);
        in.seekg(0);for(size_t i=0;i<sc::kHeaderBytes;++i)raw[i]=uint8_t((i*73+layer+expert)&255);
        in.read(reinterpret_cast<char*>(raw.data()+sc::kHeaderBytes),sc::kRawBytes-sc::kHeaderBytes);
        require(bool(in),"fixture read");
        // Independent inverse of official-scale -> RG16 mapping; this checks
        // the real-record prepass against the exact code-bearing encode path.
        auto scale=[&](size_t n){return raw[sc::kHeaderBytes+(n/16)*144+128+n%16];};
        size_t n=0;
        for(size_t group=0;group<384;++group)for(size_t col=0;col<384;++col)for(size_t row=0;row<16;++row){size_t r=group*16+row;(r&1?up:gate)[r/2*384+col]=scale(n++);}
        for(size_t group=0;group<384;++group)for(size_t col=0;col<192;++col)for(size_t row=0;row<16;++row)down[(group*16+row)*192+col]=scale(n++);
        require(n==sc::kScaleBytes,"inverse plane size");
        std::string e;size_t predicted=0,written=0;
        require(sc::encoded_size_from_scales(gate.data(),up.data(),down.data(),predicted,e),e);
        require(sc::encode(raw.data(),raw.size(),enc.data(),enc.size(),written,e),e);
        require(written==predicted,"official scale prepass mismatch");
        require(sc::validate(enc.data(),written,e),e);
        require(sc::decode_raw(enc.data(),written,back.data(),back.size(),e),e);
        require(raw==back,"exact native record mismatch");
        require(!sc::encode(raw.data(),raw.size(),enc.data(),written-256,written,e),"undersized slot accepted");
        require(sc::encode(raw.data(),raw.size(),enc.data(),enc.size(),written,e),e);
        auto corrupt=enc;
        corrupt[sc::kHeaderBytes]^=1;
        require(!sc::validate(corrupt.data(),written,e),"bad magic accepted");
        corrupt=enc;set32(corrupt.data()+sc::kHeaderBytes+4,2);
        require(!sc::validate(corrupt.data(),written,e),"bad version accepted");
        corrupt=enc;set32(corrupt.data()+sc::kHeaderBytes+12+4,0);
        require(!sc::validate(corrupt.data(),written,e),"non-increasing offsets accepted");
        corrupt=enc;set32(corrupt.data()+sc::kHeaderBytes+12+sc::kBlocks*4,0xffffffffu);
        require(!sc::validate(corrupt.data(),written,e),"out of bounds offset accepted");
        corrupt=enc;corrupt[sc::kHeaderBytes+232]=1;
        require(!sc::validate(corrupt.data(),written,e),"reserved bytes accepted");
        require(!sc::validate(enc.data(),written-1,e),"truncated record accepted");
        corrupt=enc;std::memset(corrupt.data()+sc::kHeaderBytes+sc::kDescriptorBytes+sc::kCodeBytes,0,4);
        require(!sc::decode_raw(corrupt.data(),written,back.data(),back.size(),e),"corrupt compressed block accepted");
        require(!sc::decode_raw(enc.data(),written,back.data(),back.size()-1,e),"short raw destination accepted");
        std::cout<<path<<" exact; encoded="<<written<<'\n';
    }
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
