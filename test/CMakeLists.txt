add_library(entt_cuda_api_contract OBJECT api_contract.cu dependency_contract.cu)
target_link_libraries(entt_cuda_api_contract PRIVATE EnTT::Cuda)

add_executable(entt_cuda_host_contract host_contract.cpp)
target_link_libraries(entt_cuda_host_contract PRIVATE EnTT::Cuda)
add_test(NAME entt_cuda.host_contract COMMAND entt_cuda_host_contract)
set_tests_properties(entt_cuda.host_contract PROPERTIES LABELS "host;contract")

add_executable(entt_cuda_toolchain toolchain.cu)
target_link_libraries(entt_cuda_toolchain PRIVATE EnTT::Cuda)
target_compile_definitions(entt_cuda_toolchain PRIVATE "$<$<BOOL:${ENTT_CUDA_REQUIRE_GPU}>:ENTT_CUDA_REQUIRE_GPU>")
set_target_properties(entt_cuda_toolchain PROPERTIES CUDA_STANDARD_REQUIRED YES CXX_STANDARD_REQUIRED YES)
add_test(NAME entt_cuda.toolchain COMMAND entt_cuda_toolchain)
set_tests_properties(entt_cuda.toolchain PROPERTIES SKIP_RETURN_CODE 77 LABELS "cuda;runtime")
