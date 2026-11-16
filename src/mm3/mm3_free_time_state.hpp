#pragma once
#include <cstdint>
#include <vector>

namespace mm3 {
struct FreeTimePrice {double value[2][3]{};std::uint64_t samples[2][3]{};};
// All policy inputs are checkpointed. rate_rows is a process-local observation
// anchor and is rebased when restoring; clock_rows and last_move are the logical
// policy clock, so restoring at another generated-row count preserves ages.
struct FreeTimeState {
    std::vector<FreeTimePrice> layer;
    FreeTimePrice global;
    std::vector<std::uint64_t> last_move;
    double rate[3]{},delay_rows[2]{},price_ms[2]{0.1,0.1};
    std::uint64_t price_samples[2]{},clock_rows=1,rate_rows=0;
};
} // namespace mm3
