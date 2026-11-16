// CPU-only: scheduling a real primary locate traversal before bookkeeping must
// preserve services, physical-frame order, unique copies and failure boundaries.
#include "mm3_upload_first.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <tuple>
#include <vector>
namespace uf=mm3::upload_first;
using Service=uf::Service;
static void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
struct Slot {int expert=-1,age=0;bool reserved=false,pending=false,usable=true,busy=false;};
struct State {
    std::array<bool,128> resident{},helper{},third{},stage{};
    std::vector<Slot> slots;
    bool lane=true;
    int fail_expert=-1;
    std::vector<std::pair<int,int>> copies;
    std::vector<int> locate_order;
    bool cached(int expert)const {
        for(const auto& s:slots)if(s.usable&&s.expert==expert)return true;
        return false;
    }
    bool cpu(int expert)const {return !helper[expert]&&!resident[expert]&&!cached(expert);}
    bool third_serves(int expert)const {return third[expert]&&cpu(expert);}
    void protect(const std::vector<int>& picks) {
        for(auto& s:slots)s.busy=s.usable&&std::find(picks.begin(),picks.end(),s.expert)!=picks.end();
    }
    bool locate(int expert,unsigned long long& pointer,bool& copied) {
        locate_order.push_back(expert);copied=false;
        if(expert==fail_expert)return false;
        if(resident[expert]){pointer=0x100000ULL+expert*0x1000ULL;return true;}
        int at=-1;
        for(size_t j=0;j<slots.size();++j)if(slots[j].usable&&slots[j].expert==expert)at=int(j);
        if(at<0) {
            for(size_t j=0;j<slots.size();++j){const auto& s=slots[j];
                if(!s.reserved&&!s.pending&&s.usable&&!s.busy&&(at<0||s.age<slots[at].age))at=int(j);
            }
            if(at<0)return false;
            slots[at].expert=expert;copies.push_back({expert,at});copied=true;
        }
        slots[at].busy=true;slots[at].age=23;
        pointer=0x200000ULL+at*0x1000ULL;return true;
    }
};
// Independent serial baseline matching the existing bookkeeping loop's order.
// It calls locate exactly once at each actual primary pick.
static bool baseline(State& state,int rows,int k,const std::vector<int>& ids,uf::Plan& out) {
    out=uf::Plan{};
    for(int t=0;t<rows;++t){int third_columns=0;
        for(int i=0;i<k;++i){const int x=ids[t*k+i],at=t*8+i;
            if(state.helper[x]){out.service[at]=Service::helper;continue;}
            if(state.third_serves(x)&&third_columns<4){++third_columns;out.service[at]=Service::third;continue;}
            if(state.lane&&state.cpu(x)&&!state.stage[x]){out.service[at]=Service::cpu;continue;}
            out.service[at]=Service::primary;
            if(!state.locate(x,out.pointer[at],out.copied[at]))return false;
        }
    }
    out.complete=true;return true;
}
static uf::Plan check(State original,int rows,int k,const std::vector<int>& ids,bool success=true) {
    original.protect(ids);State old=original,early=original;uf::Plan expected{},got{};
    const bool old_ok=baseline(old,rows,k,ids,expected);
    const bool new_ok=uf::prepare(rows,k,ids.data(),got,
        [&](int e){return early.helper[e];},[&](int e){return early.third_serves(e);},
        [&](int e){return early.lane&&early.cpu(e)&&!early.stage[e];},
        [&](int e,unsigned long long& p,bool& copied){return early.locate(e,p,copied);});
    need(old_ok==success&&new_ok==old_ok,"success/failure changed");
    need(got.complete==old_ok,"partial plan published as complete");
    need(got.service==expected.service&&got.pointer==expected.pointer&&got.copied==expected.copied,"pick service/frame/fresh flag changed");
    need(old.copies==early.copies&&old.locate_order==early.locate_order,"copy/locate traversal order changed");
    for(size_t j=0;j<old.slots.size();++j)
        need(std::tie(old.slots[j].expert,old.slots[j].age,old.slots[j].busy)==
             std::tie(early.slots[j].expert,early.slots[j].age,early.slots[j].busy),"frame identity/LRU/lease changed");
    return got;
}
int main(){
    try {
        State mixed;mixed.slots={{8,0},{-1,0},{9,0}};mixed.helper[1]=true;mixed.resident[2]=true;mixed.third[3]=true;mixed.stage[4]=true;
        auto p=check(mixed,2,4,{1,2,3,4,4,5,8,3});
        need(p.service[0]==Service::helper&&p.service[1]==Service::primary&&p.service[2]==Service::third&&p.copied[3],"mixed ownership routes");
        need(!p.copied[8]&&p.service[9]==Service::cpu,"repeat upload/CPU choice");
        // A selected cached frame is protected even if its first pick is later.
        State reused;reused.lane=false;reused.slots={{5,-99},{-1,0}};
        p=check(reused,1,3,{6,5,6});
        need(p.pointer[0]==p.pointer[2]&&p.pointer[1]!=p.pointer[0]&&p.copied[0]&&!p.copied[2],"selected frame overwritten or duplicate copy");
        // Mutable third predicate: an overflow staged on row1 must not rewrite
        // row0's earlier third selection when bookkeeping consumes the plan.
        State mutable_route;mutable_route.lane=false;mutable_route.slots.resize(3);
        for(int e=0;e<6;++e)mutable_route.third[e]=true;
        p=check(mutable_route,2,5,{4,0,1,2,5,0,1,2,3,4});
        need(p.service[0]==Service::third&&p.service[4]==Service::primary&&p.service[12]==Service::primary,"third capacity/earlier service changed");
        State blocked;blocked.lane=false;blocked.slots={{-1,0,true},{-1,0,false,true},{9,5},{-1,-2}};
        p=check(blocked,1,2,{6,7});
        need(p.pointer[0]==0x203000ULL&&p.pointer[1]==0x202000ULL,"reserved/pending frame or tie break changed");
        State full;full.lane=false;full.slots.resize(4);
        std::vector<int> ids;for(int t=0;t<16;++t)for(int i=0;i<4;++i)ids.push_back(i);
        check(full,16,4,ids);
        State failed;failed.lane=false;failed.slots.resize(3);failed.fail_expert=7;
        check(failed,1,3,{6,7,8},false);
        State noframe;noframe.lane=false;noframe.slots={{5,0}};
        check(noframe,1,2,{5,6},false);
        for(int rows:{0,17}){uf::Plan invalid{};int calls=0;
            need(!uf::prepare(rows,4,ids.data(),invalid,[](int){return false;},[](int){return false;},[](int){return false;},
                [&](int,unsigned long long&,bool&){++calls;return true;})&&!invalid.complete&&calls==0,"bad geometry made side effects");
        }
        std::puts("upload_first PASS: selected services/frame/LRU/copy order, repeats, mutable third predicate, reservations,16 rows and failure boundaries");
        return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"upload_first: %s\n",e.what());return 1;}
}
