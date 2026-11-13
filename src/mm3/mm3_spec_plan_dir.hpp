#pragma once
// The planner tables folder of an engine start: a request's spec_plan_file names a table there by its bare file name
// (mm3_main.cpp plan_file). It is the folder of the start's first --spec-plan table as the arguments name it (a relative
// path from the working folder), resolved by std::filesystem::canonical: links and ".." in order, not lexically first.
// No folder is written into the code. serve/server.py spec_plan_dir derives the same folder from its config's args:
// serve/test_server.py runs both on the same arguments in the same working folder (this through
// tests/mm3_spec_plan_dir_test.cpp) and requires the same folder.
#include <cstddef>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace mm3 {

// args: a start's arguments after the program's name. The first "--spec-plan" followed by a value names the table.
// "" without one (or with only a trailing "--spec-plan"); "" and err set when that table's folder cannot be resolved.
inline std::string spec_plan_dir(const std::vector<std::string>& args, std::string& err) {
    err.clear();
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] != "--spec-plan") continue;
        std::error_code ec;
        const std::filesystem::path table = std::filesystem::absolute(args[i + 1], ec);
        const std::filesystem::path folder = ec ? std::filesystem::path() : std::filesystem::canonical(table.parent_path(), ec);
        if (ec) {
            err = args[i + 1] + ": " + ec.message();
            return "";
        }
        return folder.string();
    }
    return "";
}

}  // namespace mm3
