// src/mm3/mm3_env.hpp - maya-mm3's environment settings: MAYA_<name> first (the other engines' prefix), then <name>
// itself (the MM3_* names #477 used, kept as aliases).  The switches are listed in mm3_env_switches.hpp; main() prints
// the set ones and refuses an unknown one at the start.
#pragma once

#include <cstdlib>
#include <string>

namespace mm3 {

inline const char* env(const char* name) {
    const std::string m = std::string("MAYA_") + name;
    if (const char* v = std::getenv(m.c_str())) return v;
    return std::getenv(name);
}

// A switch's value kept as it was read: callers hold it in a function-local static, so a switch is read once (at the
// load or the start of serving, wherever its first use is) and never per request, forward or layer.  maya: through
// env() (MAYA_<name> first), else the name alone.
struct EnvValue {
    bool set = false;
    std::string value;
    const char* c_str() const { return set ? value.c_str() : nullptr; }
};
inline EnvValue env_value(const char* name, bool maya = true) {
    const char* v = maya ? env(name) : std::getenv(name);
    EnvValue out;
    if (v != nullptr) {
        out.set = true;
        out.value = v;
    }
    return out;
}

}  // namespace mm3
