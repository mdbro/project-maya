#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mm3 {
// One producer owns this metadata. Workers receive immutable leased addresses
// and publish completion only; they never mutate the registry. The owner retires
// CPU/GPU readers before changing a canonical pointer or releasing a lease.
class SharedFrames {
public:
    struct Frame {long long key=-1;std::uint64_t generation=1;bool leased=false;};
    struct Lease {int frame=-1;long long key=-1;std::uint64_t generation=0;explicit operator bool()const{return frame>=0;}};
    void initialize(void* base,std::size_t bytes,std::size_t stride) {
        base_=static_cast<std::uint8_t*>(base);stride_=stride;
        frames_.assign(stride?bytes/stride:0,Frame{});
    }
    int index(const void* address)const {
        const auto p=reinterpret_cast<std::uintptr_t>(address),b=reinterpret_cast<std::uintptr_t>(base_);
        if(!base_ || p<b || !stride_ || (p-b)%stride_ || (p-b)/stride_>=frames_.size())return -1;
        return int((p-b)/stride_);
    }
    std::uint8_t* address(int i)const {return i>=0 && std::size_t(i)<frames_.size()?base_+std::size_t(i)*stride_:nullptr;}
    std::size_t size()const {return frames_.size();}
    const Frame& at(int i)const {return frames_.at(std::size_t(i));}
    bool assign(int i,long long key,std::string& error) {
        if(i<0 || std::size_t(i)>=frames_.size() || frames_[std::size_t(i)].leased) {
            error="shared frame assignment would replace a lease";return false;
        }
        auto& f=frames_[std::size_t(i)];
        if(f.key!=key) {f.key=key;++f.generation;}
        return true;
    }
    Lease take(int i,long long expected) {
        if(i<0 || std::size_t(i)>=frames_.size())return {};
        auto& f=frames_[std::size_t(i)];
        if(f.leased || f.key!=expected)return {};
        f.leased=true;return {i,f.key,f.generation};
    }
    Lease take_free() {
        for(std::size_t i=0;i<frames_.size();++i)if(frames_[i].key==-1 && !frames_[i].leased)return take(int(i),-1);
        return {};
    }
    bool current(const Lease& lease)const {
        return lease.frame>=0 && std::size_t(lease.frame)<frames_.size() &&
            frames_[std::size_t(lease.frame)].leased && frames_[std::size_t(lease.frame)].key==lease.key &&
            frames_[std::size_t(lease.frame)].generation==lease.generation;
    }
    bool finish(const Lease& lease,long long key,std::string& error) {
        if(!current(lease)) {error="stale shared-frame lease";return false;}
        auto& f=frames_[std::size_t(lease.frame)];f.key=key;f.leased=false;++f.generation;return true;
    }
    bool busy()const {for(const auto& f:frames_)if(f.leased)return true;return false;}
    std::vector<long long> roster()const {std::vector<long long> out;for(const auto& f:frames_)out.push_back(f.key);return out;}
private:
    std::uint8_t* base_=nullptr;
    std::size_t stride_=0;
    std::vector<Frame> frames_;
};
} // namespace mm3
