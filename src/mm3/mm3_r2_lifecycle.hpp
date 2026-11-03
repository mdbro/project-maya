#pragma once
#include <atomic>
#include <cstddef>
#include <exception>
#include <string>

// Shared production transitions, parameterized only at the CUDA submission/join
// boundary so controlled failures can exercise the real publication rules.
namespace mm3::r2 {
// Guard the whole model while native execution can submit or retire work.
// The caller performs argument and normal-rejection checks before entering.
// False and exceptions are sticky even if a later join would succeed.
template<class Work>
bool execution_boundary(bool native, std::atomic<bool>& uncertain, Work&& work) {
    struct Guard {
        bool native; std::atomic<bool>& uncertain; bool completed=false;
        ~Guard() noexcept { if(native && !completed)uncertain.store(true); }
    } guard{native,uncertain};
    const bool ok=work();
    guard.completed=ok;
    return ok;
}
inline bool retirement_may_begin(const std::atomic<bool>& uncertain) noexcept { return !uncertain.load(); }
template<class Owner, class Pending, class Submit>
bool foreground_issue(Owner& owner, Pending& pending, std::string& sticky,
                      std::string& error, Submit submit) {
    if (!sticky.empty()) { error=sticky; return false; }
    owner=-1;
    if (!submit(error)) { sticky=error; return false; }
    pending=1;
    return true;
}
enum class Completion { ready, not_ready, failed };
// A current-forward foreground copy can be consumed on its ordered stream.
// At later boundaries it remains reserved and invisible until completion.
template<class Clock>
bool foreground_usable(bool pending, Clock last_used, Clock current) {
    return !pending || last_used == current;
}
inline bool snapshot_visible(bool foreground_pending) { return !foreground_pending; }
template<class Clock>
bool promotion_slot_available(bool foreground_pending, bool promotion_pending, Clock last_used, Clock current) {
    return !foreground_pending && !promotion_pending && last_used != current;
}
template<class Pending, class Owners, class Query>
bool foreground_retire(Pending& pending, Owners& owners, std::string& sticky,
                       std::string& error, Query query) {
    if (!sticky.empty()) { error=sticky; return false; }
    for (size_t s=0;s<pending.size();++s) if (pending[s]) {
        const Completion status=query(s,error);
        if (status == Completion::failed) { owners[s]=-1; sticky=error; return false; }
        if (status == Completion::ready) pending[s]=0;
    }
    return true;
}
template<class Queue, class Issued, class Submit>
bool issue(Queue& queue, Issued& issued, std::string& sticky,
           std::string& error, Submit submit) {
    if (!sticky.empty()) { error = sticky; return false; }
    for (const auto& q : queue) {
        issued[static_cast<size_t>(q.slot)] = 0;
        if (!submit(q, error)) {
            sticky = error;
            queue.clear();
            return false;
        }
        issued[static_cast<size_t>(q.slot)] = 1;
    }
    queue.clear();
    return true;
}
template<class Pending, class Issued, class Owners, class Used, class Clock, class Query>
bool publish(Pending& pending, Issued& issued, Owners& owners, Used& used,
             int& n_pending, Clock now, std::string& sticky,
             std::string& error, Query query) {
    if (!sticky.empty()) { error = sticky; return false; }
    for (size_t s=0; s<pending.size(); ++s) {
        if (pending[s]<0 || !issued[s]) continue;
        const Completion status=query(s,error);
        if (status==Completion::not_ready) continue;
        if (status==Completion::failed) { sticky=error; return false; }
        owners[s]=pending[s]; pending[s]=-1; issued[s]=0; used[s]=now; --n_pending;
    }
    return true;
}
template<class Devices, class Join>
bool retire(bool uncertain, Devices& devices, Join join) {
    bool ok=!uncertain;
    for (auto& d : devices) if (!join(d)) ok=false;
    return ok;
}
template<class Work, class Report>
void guarded_worker(Work work, Report report) noexcept {
    try { work(); }
    catch (const std::exception& e) { report(e.what()); }
    catch (...) { report("unknown worker failure"); }
}
}
