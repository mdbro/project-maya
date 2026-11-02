// Included inside Model::Impl. One serialized per-layer reader lease. The
// portable table and source tensors remain owned by the Model through finish.
bool helper_queue_prepare(std::string& err) {
    if(helper_queue)return true;
    if(!native || !helper || D.size()!=1 || o.seqs!=1 || o.tp || o.native_helper<0) {
        err="queued helper requires single-sequence native primary and helper";return false;
    }
    cudaDeviceProp p{},h{};
    CK(cudaGetDeviceProperties(&p,D[0].id));CK(cudaGetDeviceProperties(&h,o.native_helper));
    if(p.major!=12 || p.minor!=0 || h.major!=8 || h.minor!=6 ||
       !std::strstr(p.name,"RTX 5090") || !std::strstr(h.name,"RTX 3090")) {
        err="queued helper requires RTX 5090 primary and RTX 3090 helper";return false;
    }
    CK(cudaSetDevice(D[0].id));
    if(!helper_queue_table_h)CK(cudaHostAlloc(&helper_queue_table_h,g.n_exp*sizeof(nf::QueueRecord),cudaHostAllocPortable));
    if(!helper_queue_table_d)CK(cudaMalloc(&helper_queue_table_d,g.n_exp*sizeof(nf::QueueRecord)));
    if(!helper_queue_picks_d)CK(cudaMalloc(&helper_queue_picks_d,kWin*g.k_exp*sizeof(nf::QueuePick)));
    try {helper_queue=std::make_unique<nf::DeviceQueue>(D[0].id,o.native_helper,kWin,2,1000000000ULL);}
    catch(const std::exception& e){err=std::string("queued helper preparation: ")+e.what();return false;}
    const auto b=helper_queue->bytes();
    info.push_back("queued helper reserved: "+std::to_string(b.helper_device)+" helper device bytes, "+
                   std::to_string(b.pinned)+" mapped pinned bytes; primary table and picks "+
                   std::to_string(g.n_exp*sizeof(nf::QueueRecord)+kWin*g.k_exp*sizeof(nf::QueuePick))+" bytes");
    return true;
}
bool helper_queue_launch_picks(Dev& d,int T,cudaStream_t stream,std::string& err) {
    helper_queue_detail::picks<<<(T*g.k_exp+127)/128,128,0,stream>>>(d.top,d.w,helper_queue_table_d,helper_queue_picks_d,T*g.k_exp,g.n_exp,g.scale);
    CK(cudaGetLastError());return true;
}
#include "mm3_native_helper_queue_owner.inc"
