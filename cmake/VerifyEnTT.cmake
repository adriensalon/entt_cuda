# Keep the compatibility reference independent of installed EnTT packages.
set(ENTT_CUDA_ENTT_REVISION "1dcd7aab6548786f7672ff578639188bd2f0d837")
set(entt_cuda_reference "${CMAKE_CURRENT_LIST_DIR}/../external/entt")

if(NOT EXISTS "${entt_cuda_reference}/src/entt/entity/registry.hpp")
    message(FATAL_ERROR "Missing EnTT reference. Run git submodule update --init external/entt.")
endif()

find_package(Git QUIET)

if(GIT_FOUND AND EXISTS "${entt_cuda_reference}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${entt_cuda_reference}" rev-parse HEAD
        OUTPUT_VARIABLE entt_cuda_revision
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE entt_cuda_git_result
    )

    if(NOT entt_cuda_git_result EQUAL 0 OR NOT entt_cuda_revision STREQUAL ENTT_CUDA_ENTT_REVISION)
        message(FATAL_ERROR "EnTT must be pinned to ${ENTT_CUDA_ENTT_REVISION}; found ${entt_cuda_revision}.")
    endif()
endif()

# Normalize checkout line endings before hashing; also works in source archives.
file(READ "${entt_cuda_reference}/src/entt/entity/registry.hpp" entt_cuda_registry_source)
string(REPLACE "\r\n" "\n" entt_cuda_registry_source "${entt_cuda_registry_source}")
string(SHA256 entt_cuda_registry_hash "${entt_cuda_registry_source}")
file(READ "${entt_cuda_reference}/src/entt/config/version.h" entt_cuda_version_source)
string(REPLACE "\r\n" "\n" entt_cuda_version_source "${entt_cuda_version_source}")
string(SHA256 entt_cuda_version_hash "${entt_cuda_version_source}")

if(NOT entt_cuda_registry_hash STREQUAL "90b70d0b483ad2d369c3a802eafcdb5bd190c0e04ee8acfcecd5b75f18134905"
   OR NOT entt_cuda_version_hash STREQUAL "c80e21d32e423f1d550e8dc3e2d64e099235d2f0b4ac636e273b1d5092961fd3")
    message(FATAL_ERROR "The pinned EnTT registry/version differs from the phase-0 API audit.")
endif()
