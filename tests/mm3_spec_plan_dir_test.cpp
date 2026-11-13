// CPU-only: the planner tables folder the engine derives from its start's arguments (src/mm3/mm3_spec_plan_dir.hpp).
// With arguments it prints the folder it derives from them - "none" without a --spec-plan, "error: ..." when the folder
// cannot be resolved - so serve/test_server.py can require that serve/server.py spec_plan_dir derives the same folder
// from the same arguments in the same working folder. Without arguments it checks itself.
//     g++ -std=c++17 tests/mm3_spec_plan_dir_test.cpp -o mm3_spec_plan_dir_test && ./mm3_spec_plan_dir_test
#include "../src/mm3/mm3_spec_plan_dir.hpp"
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

int main(int argc, char** argv) {
    std::string err;
    if (argc > 1) {
        const std::string dir = mm3::spec_plan_dir(std::vector<std::string>(argv + 1, argv + argc), err);
        if (!err.empty()) std::printf("error: %s\n", err.c_str());
        else std::printf("%s\n", dir.empty() ? "none" : dir.c_str());
        return 0;
    }
    namespace fs = std::filesystem;
    const fs::path root = fs::canonical(fs::temp_directory_path()) / ("mm3-spec-plan-dir-" + std::to_string(getpid()));
    fs::create_directories(root / "first");
    fs::create_directories(root / "second");
    // none: no --spec-plan, or only a trailing one without a value
    assert(mm3::spec_plan_dir({}, err).empty() && err.empty());
    assert(mm3::spec_plan_dir({"--pack", "x"}, err).empty() && err.empty());
    assert(mm3::spec_plan_dir({"--pack", "x", "--spec-plan"}, err).empty() && err.empty());
    // the first table's folder, of two
    assert(mm3::spec_plan_dir({"--spec-plan", (root / "first" / "a.txt").string(), "--spec-plan",
                               (root / "second" / "b.txt").string()}, err) == (root / "first").string() && err.empty());
    // a relative path: from the working folder
    fs::current_path(root);
    assert(mm3::spec_plan_dir({"--spec-plan", "second/b.txt"}, err) == (root / "second").string() && err.empty());
    // a folder that does not exist: an error, not a folder
    assert(mm3::spec_plan_dir({"--spec-plan", (root / "missing" / "a.txt").string()}, err).empty() && !err.empty());
    fs::current_path(fs::temp_directory_path());
    fs::remove_all(root);
    std::printf("mm3_spec_plan_dir_test: ok\n");
    return 0;
}
