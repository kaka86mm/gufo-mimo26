# MiMo-V2.6 (mimo2) model package. The scalar oracle is built only by
# explicit validation targets, not by production or routine checks.
option(GUFO_ENABLE_TP2_RDMA
  "Build the mimo2 TP=2 native RDMA adapter" OFF)

add_library(gufo_mimo2 STATIC
  chat_template.cpp
  config.cpp
  tokenizer.cpp
  weights.cpp
  distributed/rdma_gid.cpp
  distributed/tp_partition.cpp)

target_include_directories(gufo_mimo2 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_mimo2 PUBLIC gufo_core Threads::Threads)

if (GUFO_ENABLE_TP2_RDMA)
  if (NOT ENGINE_ENABLE_HIP)
    message(FATAL_ERROR
      "GUFO_ENABLE_TP2_RDMA requires ENGINE_ENABLE_HIP")
  endif()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(IBVERBS REQUIRED IMPORTED_TARGET libibverbs)
  add_library(gufo_mimo2_rdma STATIC
    kernels/rocm/verbs.cpp)
  target_compile_features(gufo_mimo2_rdma PRIVATE cxx_std_20)
  target_include_directories(gufo_mimo2_rdma PRIVATE
    ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_mimo2_rdma PRIVATE
    PkgConfig::IBVERBS Threads::Threads ${CMAKE_DL_LIBS} hip::host)
  target_link_libraries(gufo_mimo2 PRIVATE
    gufo_mimo2_rdma)
  target_compile_definitions(gufo_mimo2 PUBLIC
    GUFO_ENABLE_TP2_RDMA=1)
endif()

add_library(gufo_mimo2_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_mimo2_reference PUBLIC
  gufo_mimo2)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

# The ROCm runtime sources land as the executor port completes; flip
# GUFO_MIMO2_HIP=ON to compile them into the model package.
option(GUFO_MIMO2_HIP "Build the mimo2 ROCm runtime" OFF)

if (ENGINE_ENABLE_HIP AND GUFO_MIMO2_HIP)
  set(QFN_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
  set(QFN_MMQ_DIR "${QFN_ROCM_DIR}/mmq")

  # HIP quantized matrix kernels adapted from llama.cpp, isolated in the
  # qfn_mmq namespace. The adapter retains its C++17 dialect.
  set(QFN_MMQ_SOURCES
    ${QFN_MMQ_DIR}/mimo2_ggml_stubs.hip.cpp
    ${QFN_MMQ_DIR}/mimo2_mmq.hip.cpp
    ${QFN_MMQ_DIR}/mimo2_mmq_d2r.hip.cpp
    ${QFN_MMQ_DIR}/mmid.hip.cpp
    ${QFN_MMQ_DIR}/mmvq.hip.cpp
    ${QFN_MMQ_DIR}/quantize.hip.cpp)

  add_library(gufo_mimo2_mmq STATIC ${QFN_MMQ_SOURCES})
  set_source_files_properties(${QFN_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_mimo2_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_mimo2_mmq PRIVATE
    GGML_USE_HIP
    GGML_HIP_NO_VMM
    MIMO2_ROCM_WMMA_MMQ_Y=64
    # The vendored glue collides with deepseek_v4_flash's copy of the same
    # upstream when both archives land in one binary; rename ours out of the
    # way. Entry points are already mimo2_-prefixed.
    ggml_time_us=mimo2_ggml_time_us
    get_mmvq_mmid_max_batch=mimo2_get_mmvq_mmid_max_batch
    iq2_xxs_aligned_derepack_kernel=mimo2_iq2_xxs_aligned_derepack_kernel
    q8_0_aligned_dense_vec_kernel=mimo2_q8_0_aligned_dense_vec_kernel)
  target_compile_options(gufo_mimo2_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_mimo2_mmq SYSTEM PUBLIC ${QFN_MMQ_DIR})
  target_link_libraries(gufo_mimo2_mmq PUBLIC hip::host roc::hipblas)

  set(QFN_HIP_SOURCES
    ${QFN_ROCM_DIR}/kernels.hip.cpp
    ${QFN_ROCM_DIR}/dflash_v1_kernels.hip.cpp
    ${QFN_ROCM_DIR}/w8a8_wave64.hip
    ${QFN_ROCM_DIR}/blaslt.cpp
    ${QFN_ROCM_DIR}/device_model.cpp
    ${QFN_ROCM_DIR}/executor.cpp
    ${QFN_ROCM_DIR}/batch.cpp
    dflash_v1.cpp
    dflash_v1_target.cpp
    continuation_adapter.cpp
    engine.cpp)
  target_sources(gufo_mimo2 PRIVATE ${QFN_HIP_SOURCES})
  set_source_files_properties(${QFN_ROCM_DIR}/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${QFN_ROCM_DIR}/dflash_v1_kernels.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${QFN_ROCM_DIR}/w8a8_wave64.hip
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  target_compile_definitions(gufo_mimo2 PRIVATE QFN_ROCM_BUILD=1)
  target_compile_options(gufo_mimo2 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_mimo2 SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
  target_link_libraries(gufo_mimo2 PUBLIC
    gufo_mimo2_mmq hip::host roc::hipblas roc::hipblaslt)

  # DFlash v1 drafter bring-up probe: explicit target so routine builds stay
  # unchanged; it loads the real sidecar and needs a GPU.
  add_executable(mimo2_dflash_probe EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tests/models/mimo2/dflash_probe.cpp)
  target_link_libraries(mimo2_dflash_probe PRIVATE gufo_mimo2 hip::host)
endif()
