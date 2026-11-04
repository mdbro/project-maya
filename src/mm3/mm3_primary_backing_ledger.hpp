#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mm3::primary_backing_ledger {
// At an already joined publication, transfer the two physical HOST copies
// between exclusive host ownership and the separate primary-copy map.
inline bool rotate(const uint8_t*& incoming_host, uint8_t*& incoming_backing,
                   const uint8_t*& victim_host, uint8_t*& victim_backing) {
    if(!incoming_host || incoming_backing || victim_host || !victim_backing ||
       incoming_host==victim_backing)return false;
    incoming_backing=const_cast<uint8_t*>(incoming_host);
    incoming_host=nullptr;victim_host=victim_backing;victim_backing=nullptr;
    return true;
}
// After all readers retire, replace any HOST owner in the soon-to-be-freed
// bank with an otherwise discarded primary backing frame outside that bank.
inline bool repatriate(const std::vector<const uint8_t**>& hosts,
                       const std::vector<uint8_t*>& free_frames,size_t bytes) {
    if(hosts.size()!=free_frames.size() || !bytes)return false;
    std::vector<uintptr_t> all;all.reserve(hosts.size()*2);
    for(size_t i=0;i<hosts.size();++i) {
        if(!hosts[i] || !*hosts[i] || !free_frames[i])return false;
        all.push_back(uintptr_t(*hosts[i]));all.push_back(uintptr_t(free_frames[i]));
    }
    std::sort(all.begin(),all.end());
    for(size_t i=1;i<all.size();++i)if(all[i]-all[i-1]<bytes)return false;
    for(size_t i=0;i<hosts.size();++i) {
        std::memcpy(free_frames[i],*hosts[i],bytes);*hosts[i]=free_frames[i];
    }
    return true;
}
}
