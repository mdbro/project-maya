// CPU-only execution of the literal production owner methods. CUDA transport
// and DeviceQueue are controllable fakes; the real device kernels and mapped
// cross-GPU protocol have separate focused fixtures.
#include "cuda_runtime_api.h"
#include "mm3_resident_loader.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define CK(call) do {auto e_=(call);if(e_!=cudaSuccess){err=std::string(#call)+": "+cudaGetErrorString(e_);return false;}} while(0)
constexpr int kWin=16;
namespace nf {
struct QueuePick {int slot;float weight;};
struct QueueRecord {std::uint64_t helper_record;float multiplier[3];};
struct DeviceQueue {
 bool active=false,published=false,cancelled=false,producer=false,consumer=false;
 bool fail_schedule=false,unsafe_schedule=false,fail_publish=false,fail_finish=false,cancel_retires=true;
 int columns=0,aborts=0,finishes=0;std::uint64_t seq=0;
 std::vector<QueuePick> captured_picks;std::array<QueueRecord,128> captured_table{};
 bool schedule(std::uint64_t s,int t,std::string& err){
  if(fail_schedule){active=unsafe_schedule;err="injected schedule";return false;}
  assert(!active);active=true;published=cancelled=producer=consumer=false;seq=s;columns=t;return true;
 }
 bool publish(std::uint64_t s,const QueuePick* p,const QueueRecord* table,int nr,const float*,cudaStream_t,std::string& err){
  assert(active&&s==seq&&nr==128);if(fail_publish){err="injected publish";return false;}
  published=true;captured_picks.assign(p,p+4*columns);std::copy_n(table,nr,captured_table.begin());return true;
 }
 bool abort(std::uint64_t s,std::string&){assert(active&&s==seq);++aborts;cancelled=true;return true;}
 bool finish(std::uint64_t s,float* out,std::size_t n,int timeout,std::string& err){
  assert(active&&s==seq&&timeout==10000);++finishes;
  if(cancelled&&cancel_retires)producer=consumer=true;
  if(!producer||!consumer){err="injected unknown writer";return false;}
  active=false;if(cancelled||fail_finish){err=cancelled?"aborted":"injected finish";return false;}
  std::fill_n(out,n,0.f);
  for(int t=0;t<columns;++t)for(int i=0;i<4;++i){auto p=captured_picks[t*4+i];if(p.slot>=0&&p.slot<128)out[t*4]+=p.weight;}
  return true;
 }
 bool storage_safe()const{return !active;}
};
}
struct NativeHelper {
 struct QueueIdentity {int slot=-1;float multiplier[3]{};};
 std::array<QueueIdentity,128> entries{};bool fail=false;mutable int snapshots=0;
 bool queue_snapshot(int layer,std::array<QueueIdentity,128>& out,std::string& err)const{
  assert(layer==0);++snapshots;if(fail){err="injected snapshot";return false;}out=entries;return true;
 }
 std::uintptr_t pool_identity()const{return 0x100000;}std::size_t record_stride()const{return 4096;}
};
struct Layer {std::array<bool,128> ready{};};
struct Dev {int* top=nullptr;float* w=nullptr;float* xn=nullptr;};
struct Owner {
 struct Geometry {int n_exp=128,k_exp=4,dim=4;float scale=2.f;} g;
 std::vector<Layer> L{1};std::unique_ptr<NativeHelper> helper=std::make_unique<NativeHelper>();
 std::unique_ptr<nf::DeviceQueue> helper_queue=std::make_unique<nf::DeviceQueue>();
 std::array<nf::QueueRecord,128> ht{},dt{};std::array<nf::QueuePick,4*kWin> picks{};
 nf::QueueRecord *helper_queue_table_h=ht.data(),*helper_queue_table_d=dt.data();
 nf::QueuePick* helper_queue_picks_d=picks.data();
 std::uint64_t helper_queue_next=0,helper_queue_pending=0;
 std::atomic<bool> retirement_uncertain{false};bool placement_failed=false,launch_fail=false;
 bool helper_owns(const Layer& l,int e)const{return l.ready[e];}
 bool helper_queue_launch_picks(Dev& d,int T,cudaStream_t,std::string& err){
  if(launch_fail){err="injected pick launch";return false;}
  for(int i=0;i<T*4;++i){int e=d.top[i];picks[i]=(e<0||e>=128)?nf::QueuePick{128,0}:
   nf::QueuePick{dt[e].helper_record?e:-1,d.w[i]*g.scale};}return true;
 }
#include "mm3_native_helper_queue_owner.inc"
 void ready(int e,int slot,float m){L[0].ready[e]=slot>=0;helper->entries[e]={};helper->entries[e].slot=slot;
  for(float& x:helper->entries[e].multiplier)x=m;}
};
int main(){
 std::string error;int top[16]={1,2,3,4,2,1,5,6,3,1,7,8,4,2,9,10};
 float w[16];for(int i=0;i<16;++i)w[i]=float(i+1)/32;float x[16]{},output[16]{};Dev d{top,w,x};
 {
  Owner o;o.ready(1,5,.5f);o.ready(2,7,.25f);
  assert(o.helper_queue_begin(o.L[0],d,4,nullptr,error));auto& q=*o.helper_queue;
  assert(q.captured_table[1].helper_record==0x100000+5*4096&&q.captured_table[1].multiplier[0]==.5f);
  assert(q.captured_table[2].helper_record==0x100000+7*4096&&q.captured_table[2].multiplier[2]==.25f);
  assert(q.captured_picks[0].slot==1&&q.captured_picks[1].slot==2&&q.captured_picks[2].slot==-1);
  assert(q.captured_picks[5].slot==1&&q.captured_picks[5].weight==2*w[5]);
  assert(!o.helper_queue_destroy_safe());auto old=o.ht[1].helper_record;
  // A new snapshot cannot overwrite the borrowed table while either writer is active.
  assert(!o.helper_queue_begin(o.L[0],d,4,nullptr,error)&&o.ht[1].helper_record==old&&o.helper->snapshots==1);
  q.consumer=true;assert(!o.helper_queue_destroy_safe());q.consumer=false;
  q.producer=true;assert(!o.helper_queue_destroy_safe());q.consumer=true;
  assert(o.helper_queue_finish(4,output,error)&&o.helper_queue_destroy_safe()&&output[0]==2*(w[0]+w[1]));
  // Change ownership only after the previous ticket and its producer retire.
  o.ready(1,-1,0);o.ready(3,5,.75f);assert(o.helper_queue_begin(o.L[0],d,4,nullptr,error));
  assert(q.captured_table[1].helper_record==0&&q.captured_table[3].helper_record==old&&q.captured_table[3].multiplier[0]==.75f);
  q.producer=q.consumer=true;assert(o.helper_queue_finish(4,output,error));
  o.ready(2,-1,0);o.ready(3,-1,0);assert(o.helper_queue_begin(o.L[0],d,4,nullptr,error));
  assert(std::all_of(q.captured_picks.begin(),q.captured_picks.end(),[](const auto& p){return p.slot<0;}));
  q.producer=q.consumer=true;std::fill_n(output,16,-1.f);assert(o.helper_queue_finish(4,output,error));
  assert(std::all_of(output,output+16,[](float v){return v==0;}));
 }
 {
  Owner o;o.ready(1,0,1);fake_fail_copy=true;assert(!o.helper_queue_begin(o.L[0],d,1,nullptr,error));fake_fail_copy=false;
  assert(o.helper_queue_pending&&o.helper_queue_failed_window(true)==false);
  assert(o.helper_queue->aborts==1&&o.helper_queue_destroy_safe()&&o.placement_failed&&!o.helper_queue_pending);
  assert(!o.helper_queue_begin(o.L[0],d,1,nullptr,error)); // restart required, despite safe cancellation
 }
 for(int failure=0;failure<2;++failure) {
  Owner o;o.ready(1,0,1);o.launch_fail=failure==0;o.helper_queue->fail_publish=failure==1;
  assert(!o.helper_queue_begin(o.L[0],d,1,nullptr,error)&&o.helper_queue_pending);
  assert(!o.helper_queue_failed_window(true)&&o.helper_queue->aborts==1&&o.helper_queue_destroy_safe());
  assert(o.placement_failed&&!o.helper_queue_begin(o.L[0],d,1,nullptr,error));
 }
 {
  Owner o;o.ready(1,0,1);o.helper_queue->fail_schedule=true;o.helper_queue->unsafe_schedule=true;
  assert(!o.helper_queue_begin(o.L[0],d,1,nullptr,error));
  assert(o.placement_failed&&o.retirement_uncertain.load()&&!o.helper_queue_pending&&!o.helper_queue_destroy_safe());
  assert(!o.helper_queue_begin(o.L[0],d,1,nullptr,error));
 }
 {
  Owner o;o.ready(1,0,1);assert(o.helper_queue_begin(o.L[0],d,1,nullptr,error));
  auto& q=*o.helper_queue;q.producer=q.consumer=true;q.fail_finish=true;
  assert(!o.helper_queue_finish(1,output,error));assert(o.placement_failed&&o.helper_queue_destroy_safe());
  assert(!o.helper_queue_begin(o.L[0],d,1,nullptr,error));
 }
 {
  auto* o=new Owner;o->ready(1,0,1);assert(o->helper_queue_begin(o->L[0],d,1,nullptr,error));
  o->helper_queue->producer=true;o->helper_queue->consumer=false;o->helper_queue->cancel_retires=false;
  assert(!o->helper_queue_finish(1,output,error));assert(o->retirement_uncertain.load()&&!o->helper_queue_destroy_safe());
  o->helper_queue_failed_window(true);assert(o->helper_queue_pending&&o->placement_failed);
  assert(!o->helper_queue_begin(o->L[0],d,1,nullptr,error));
  bool disposed=false,retained=false;
  mm3::r2::destroy_after_retire(o,[&]{return o->helper_queue_destroy_safe();},[&](Owner* p){disposed=true;delete p;},[&]{retained=true;});
  assert(retained&&!disposed);
  // Fake-only cleanup after the quarantine assertion; no device writer exists.
  o->helper_queue->active=false;delete o;
 }
 {
  Owner o;o.ready(1,0,1);o.helper->fail=true;
  assert(!o.helper_queue_begin(o.L[0],d,1,nullptr,error));assert(!o.helper_queue_failed_window(true)&&o.placement_failed);
  assert(o.helper_queue_destroy_safe()&&!o.helper_queue_pending);
 }
 {
  auto* o=new Owner;bool disposed=false,retained=false;
  mm3::r2::destroy_after_retire(o,[&]{return o->helper_queue_destroy_safe();},[&](Owner* p){disposed=true;delete p;},[&]{retained=true;});
  assert(disposed&&!retained);
 }
}
