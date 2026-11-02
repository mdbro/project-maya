#pragma once
#include <cstddef>
#include <exception>
#include <string>

// Shared production transitions, parameterized only at the CUDA submission/join
// boundary so controlled failures can exercise the real publication rules.
namespace mm3::r2 {
template<class Owner, class Pending, class Submit>
bool foreground_issue(Owner& owner, Pending& pending, std::string& sticky,
                      std::string& error, Submit submit) {
    if (!sticky.empty()) { error=sticky; return false; }
    owner=-1;
    if (!submit(error)) { sticky=error; return false; }
    pending=1;
    return true;
}
template<class Pending, class Owners, class Wait>
bool foreground_retire(Pending& pending, Owners& owners, std::string& sticky,
                       std::string& error, Wait wait) {
    if (!sticky.empty()) { error=sticky; return false; }
    for (size_t s=0;s<pending.size();++s) if (pending[s]) {
        if (!wait(s,error)) { owners[s]=-1; sticky=error; return false; }
        pending[s]=0;
    }
    return true;
}
template<class Queue, class Issued, class Clock, class Submit>
bool issue(Queue& queue, Issued& issued, Clock now, std::string& sticky,
           std::string& error, Submit submit) {
    if (!sticky.empty()) { error = sticky; return false; }
    for (const auto& q : queue) {
        if (!submit(q, error)) {
            sticky = error;
            queue.clear();
            return false;
        }
        issued[static_cast<size_t>(q.slot)] = now;
    }
    queue.clear();
    return true;
}
template<class Pending, class Issued, class Owners, class Used, class Clock, class Wait>
bool publish(Pending& pending, Issued& issued, Owners& owners, Used& used,
             int& n_pending, Clock now, int lag, std::string& sticky,
             std::string& error, Wait wait) {
    if (!sticky.empty()) { error = sticky; return false; }
    for (size_t s=0; s<pending.size(); ++s) {
        if (pending[s]<0 || now-issued[s]<lag) continue;
        if (!wait(s,error)) { sticky=error; return false; }
        owners[s]=pending[s]; pending[s]=-1; used[s]=now; --n_pending;
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
