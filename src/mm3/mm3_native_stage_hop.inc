    // Serial prefill/single-token stage crossing. The portable buffer is not
    // reusable until both device streams have retired. A failed submission or
    // join quarantines the entire model even if a later join might succeed.
    bool stage_serial_hop(Dev& a, Dev& b, size_t bytes, float* dst, std::string& err) {
        if(!serial_bridge || bytes>serial_bridge_bytes) {
            err="native serial bridge capacity exceeded";return false;
        }
        auto checked = [&](cudaError_t status,const char* operation) {
            if(status==cudaSuccess)return true;
            stage_poisoned=true;retirement_uncertain.store(true);
            err=std::string(operation)+": "+cudaGetErrorString(status);return false;
        };
        if(!checked(cudaSetDevice(a.id),"serial bridge source device") ||
           !checked(cudaMemcpyAsync(serial_bridge,a.h,bytes,cudaMemcpyDeviceToHost,a.s),"serial bridge D2H") ||
           !checked(cudaStreamSynchronize(a.s),"serial bridge source retirement") ||
           !checked(cudaSetDevice(b.id),"serial bridge destination device") ||
           !checked(cudaMemcpyAsync(dst,serial_bridge,bytes,cudaMemcpyHostToDevice,b.s),"serial bridge H2D") ||
           !checked(cudaStreamSynchronize(b.s),"serial bridge destination retirement"))return false;
        return true;
    }
