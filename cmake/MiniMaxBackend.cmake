# Shared, native-WSL MiniMax compute backend for the three assigned runtimes.
# This deliberately excludes mm3_main.cpp: request scheduling, pending output,
# commit/cancel and serving belong to the consuming runtime. No sources or model
# assets are downloaded by this helper.
include_guard(GLOBAL)

function(minimax_add_backend target)
  if(TARGET ${target})
    return()
  endif()
  cmake_parse_arguments(MB "" "MAYA_ROOT;SERVICE_ROOT;GGML_ROOT" "" ${ARGN})
  if(NOT MB_MAYA_ROOT)
    get_filename_component(MB_MAYA_ROOT "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
  endif()
  if(NOT MB_SERVICE_ROOT)
    get_filename_component(MB_SERVICE_ROOT "${MB_MAYA_ROOT}/../Infernix-minimax" ABSOLUTE)
  endif()
  if(NOT MB_GGML_ROOT)
    set(MB_GGML_ROOT "${MB_MAYA_ROOT}/build-minimax/_deps/strata_llamacpp-src/ggml")
  endif()
  foreach(required
      "${MB_MAYA_ROOT}/src/mm3/mm3_model.cu"
      "${MB_SERVICE_ROOT}/src/ops/minimax_nvfp4_ffn/native_fp4.cu"
      "${MB_GGML_ROOT}/CMakeLists.txt")
    if(NOT EXISTS "${required}")
      message(FATAL_ERROR "MiniMax backend requires the provisioned source: ${required}")
    endif()
  endforeach()
  find_package(CUDAToolkit REQUIRED)
  find_package(Threads REQUIRED)
  find_library(MINIMAX_BACKEND_ZSTD NAMES zstd REQUIRED)

  # Legacy-format symbols remain dependencies of the qualified model loader;
  # native MiniMax does not select those formats. Reuse the already pinned ggml.
  if(NOT TARGET ggml-base)
    set(GGML_STATIC ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${MB_GGML_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/${target}-ggml" EXCLUDE_FROM_ALL)
  endif()
  if(NOT TARGET ggml-cpu)
    message(FATAL_ERROR "MiniMax backend requires the pinned ggml CPU target")
  endif()

  set(cpu "${MB_MAYA_ROOT}/src/kernels/cpu")
  add_library(${target}_cpu STATIC
    ${cpu}/pool.cpp ${cpu}/expert_layout.cpp ${cpu}/exl3_ref.cpp
    ${cpu}/expert.cpp ${cpu}/q2_avx2.cpp ${cpu}/exl3_cpu.cpp
    ${cpu}/native_expert.cpp ${cpu}/iq_avx512.cpp ${cpu}/iq_avx2.cpp
    ${cpu}/kq_avx512.cpp ${cpu}/iq_avx512_one.cpp
    ${cpu}/glm_lane_row4.cpp ${cpu}/glm_lane_row4_avx2.cpp)
  target_compile_features(${target}_cpu PUBLIC cxx_std_20)
  target_include_directories(${target}_cpu PUBLIC "${MB_MAYA_ROOT}/include"
    PRIVATE "${MB_MAYA_ROOT}/third_party/ggml")
  target_compile_definitions(${target}_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  target_link_libraries(${target}_cpu PUBLIC ggml-cpu ggml-base Threads::Threads)
  set_source_files_properties(${cpu}/expert.cpp PROPERTIES COMPILE_OPTIONS
    "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(${cpu}/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(${cpu}/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
    "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mf16c;-mfma;-ffp-contract=off")
  set_source_files_properties(${cpu}/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS
    "-mavx2;-mfma;-mf16c;-ffp-contract=off")
  set_source_files_properties(${cpu}/kq_avx512.cpp PROPERTIES COMPILE_OPTIONS
    "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mf16c;-mfma")
  set_source_files_properties(${cpu}/iq_avx512_one.cpp PROPERTIES COMPILE_OPTIONS
    "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vbmi;-mbmi2;-mf16c;-mfma;-ffp-contract=off")
  # (upstream c5842c5: native_expert.cpp's one-token split rows call the GLM four-row lane's hooks, which only
  #  glm_lane_row4.cpp defines - as maya's own CPU library builds them, cmake/glm_dsa.d/70-glm-lane-row4.cmake)
  set_source_files_properties(${cpu}/glm_lane_row4.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off"
    COMPILE_DEFINITIONS STRATA_GLM_LANE_ROW4_KERNELS=1)
  set_source_files_properties(${cpu}/glm_lane_row4_avx2.cpp PROPERTIES COMPILE_OPTIONS
    "-mavx2;-mfma;-mf16c;-ffp-contract=off" COMPILE_DEFINITIONS STRATA_GLM_LANE_ROW4_KERNELS=1)

  set(service "${MB_SERVICE_ROOT}/src/ops/minimax_nvfp4_ffn")
  add_library(${target}_fast STATIC ${service}/record.cpp ${service}/fast_cpu.cpp
    ${service}/fast_cuda.cu ${service}/device_queue.cu)
  target_include_directories(${target}_fast PUBLIC "${MB_SERVICE_ROOT}/src")
  target_link_libraries(${target}_fast PUBLIC CUDA::cudart)
  set_target_properties(${target}_fast PROPERTIES CUDA_ARCHITECTURES "86;120"
    CXX_STANDARD 20 CUDA_STANDARD 20)
  add_library(${target}_fp4 STATIC ${service}/native_fp4.cu)
  target_include_directories(${target}_fp4 PUBLIC "${MB_SERVICE_ROOT}/src" "${MB_SERVICE_ROOT}/include")
  target_link_libraries(${target}_fp4 PUBLIC CUDA::cudart)
  set_target_properties(${target}_fp4 PROPERTIES CUDA_ARCHITECTURES "120a"
    CXX_STANDARD 20 CUDA_STANDARD 20)

  set(mm3 "${MB_MAYA_ROOT}/src/mm3")
  set(ds4 "${MB_MAYA_ROOT}/src/ds4")
  add_library(${target} STATIC
    ${mm3}/mm3_model.cu ${mm3}/mm3_cpu.cpp ${mm3}/mm3_host.cpp
    ${mm3}/mm3_exl3.cu ${mm3}/mm3_dspark.cu ${mm3}/mm3_vision.cu ${mm3}/mm3_tp.cu
    ${mm3}/mm3_gguf.cu ${mm3}/mm3_gv.cu ${mm3}/mm3_cpu_gg.cpp
    ${ds4}/ds4_iq.cu ${ds4}/ds4_lt.cu ${ds4}/ds4_iq_cpu.cpp ${ds4}/ds4_gguf_cpu.cpp
    ${mm3}/mm3_native_scale_store.cpp ${mm3}/mm3_native_scale_store.cu
    ${mm3}/mm3_native_dense.cu ${mm3}/mm3_native_experts.cu
    ${mm3}/mm3_native_cpu.cpp ${mm3}/mm3_native_helper.cu ${mm3}/mm3_native_host.cpp
    ${mm3}/mm3_resident_router.cu ${mm3}/mm3_kernels.cu)
  target_include_directories(${target} PUBLIC "${mm3}" "${MB_MAYA_ROOT}/include"
    PRIVATE "${MB_MAYA_ROOT}/third_party/ggml")
  target_compile_definitions(${target} PRIVATE MAYA_DS4_IQ_ONLY)
  target_link_libraries(${target} PUBLIC ${target}_fast ${target}_fp4 ${target}_cpu
    CUDA::cudart CUDA::cuda_driver CUDA::cublas Threads::Threads ${MINIMAX_BACKEND_ZSTD})
  set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "86;120"
    CXX_STANDARD 20 CUDA_STANDARD 20)
endfunction()
