cmake_minimum_required(VERSION 3.21)
project(zipburger_r10_gguf_probe LANGUAGES C CXX)

# Research-only. Use an isolated llama.cpp checkout at the pinned revision.
if(NOT DEFINED LLAMA_CPP_ROOT OR NOT EXISTS "${LLAMA_CPP_ROOT}/include/llama.h")
    message(FATAL_ERROR "LLAMA_CPP_ROOT must point to an existing llama.cpp source tree")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
# Ship one standalone CPU probe, not a binary tied to unavailable sibling .so files.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/upstream-llama")

add_executable(zipburger_gguf_eval_r10
    "${CMAKE_CURRENT_LIST_DIR}/../gguf_eval_r10.cpp"
)
target_include_directories(zipburger_gguf_eval_r10 PRIVATE
    "${LLAMA_CPP_ROOT}/include"
    "${LLAMA_CPP_ROOT}/ggml/include"
)
target_link_libraries(zipburger_gguf_eval_r10 PRIVATE llama ggml)

add_executable(zipburger_gguf_r10_callback_test
    "${CMAKE_CURRENT_LIST_DIR}/callback_test_r10.cpp"
)
target_include_directories(zipburger_gguf_r10_callback_test PRIVATE
    "${LLAMA_CPP_ROOT}/include"
    "${LLAMA_CPP_ROOT}/ggml/include"
)
target_link_libraries(zipburger_gguf_r10_callback_test PRIVATE llama ggml)

add_executable(zipburger_gguf_r10_graph_compute_test
    "${CMAKE_CURRENT_LIST_DIR}/graph_compute_test_r10.cpp"
)
target_include_directories(zipburger_gguf_r10_graph_compute_test PRIVATE
    "${LLAMA_CPP_ROOT}/include"
    "${LLAMA_CPP_ROOT}/ggml/include"
)
target_link_libraries(zipburger_gguf_r10_graph_compute_test PRIVATE llama ggml)

# R11 reuses the exact R10 callback implementation but holds one GGUF model
# in RAM while executing multiple strictly isolated tiny prompts sequentially.
add_executable(zipburger_gguf_batch_r11
    "${CMAKE_CURRENT_LIST_DIR}/../gguf_eval_batch_r11.cpp"
)
target_include_directories(zipburger_gguf_batch_r11 PRIVATE
    "${LLAMA_CPP_ROOT}/include"
    "${LLAMA_CPP_ROOT}/ggml/include"
)
target_link_libraries(zipburger_gguf_batch_r11 PRIVATE llama ggml)
