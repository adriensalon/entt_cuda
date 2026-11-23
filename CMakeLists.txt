cmake_minimum_required(VERSION 3.28)

project(EnTTCuda VERSION 0.1.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" OR CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.2)
    message(FATAL_ERROR "EnTTCuda currently requires NVIDIA nvcc 13.2 or newer, the phase-0 baseline. See README.md for workspace-local setup.")
endif()

include(cmake/VerifyEnTT.cmake)
find_package(CUDAToolkit 13.2 REQUIRED)

add_subdirectory(external/entt EXCLUDE_FROM_ALL)

add_library(EnTTCuda INTERFACE)
add_library(EnTT::Cuda ALIAS EnTTCuda)

target_include_directories(EnTTCuda INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(EnTTCuda INTERFACE cxx_std_20 cuda_std_20)
target_compile_options(EnTTCuda INTERFACE "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda;--expt-relaxed-constexpr>")
target_link_libraries(EnTTCuda INTERFACE EnTT::EnTT CUDA::cudart)

option(ENTT_CUDA_BUILD_TESTING "Build the CUDA registry contract and toolchain tests." ${PROJECT_IS_TOP_LEVEL})
option(ENTT_CUDA_REQUIRE_GPU "Fail CUDA runtime tests instead of skipping when no usable device is available." OFF)

if(ENTT_CUDA_BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()
