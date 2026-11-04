#pragma once
// Standalone fixtures compile production fragments outside mm3's namespace.
#include <sstream>
#include <string>
namespace capture {
inline bool detailed_enabled = false;
inline bool detailed() { return detailed_enabled; }
inline std::string quote(const std::string& value) { return value; }
inline void detail_event(const std::string&, const std::string& = {}) {}
}
