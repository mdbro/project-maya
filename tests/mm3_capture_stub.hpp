#pragma once
// Standalone fixtures compile production fragments outside mm3's namespace.
#include <cstdint>
#include <sstream>
#include <string>
namespace capture {
inline bool detailed_enabled = false;
inline bool detailed() { return detailed_enabled; }
inline std::string quote(const std::string& value) { return value; }
inline void detail_event(const std::string&, const std::string& = {}) {}
// Fragments report to the light recorder through free functions; fixtures ignore them.
namespace light {
enum HelperPhase : int32_t { kHelperLease = 1, kHelperWithdrawn = 2, kHelperPublished = 3, kHelperRetired = 4 };
inline uint64_t stamp() { return 0; }
template <class... Args> inline void foreground_retire(Args&&...) {}
template <class... Args> inline void primary_issue(Args&&...) {}
template <class... Args> inline void primary_publish(Args&&...) {}
template <class... Args> inline void helper_event(Args&&...) {}
}
}
