set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
cmake_minimum_required(VERSION 3.22)

include(FetchContent)

project(gemma_cpp_js)

# --- Define the Environment String Option ---
set(ENVIRONMENT "" CACHE STRING "Emscripten environment to build for (e.g., web, node, worker)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options(-sMEMORY64=1 -msimd128)
add_link_options(-sMEMORY64=1)

add_compile_options(-pthread)
add_link_options(-pthread)

FetchContent_Declare(gemma GIT_REPOSITORY https://github.com/brendandahl/gemma.cpp.git GIT_TAG 94648da6f24d7615f7aa078d1b29288a7beb26eb)
FetchContent_MakeAvailable(gemma)

add_executable(gemma_cpp_js src/bindings.cpp)
em_link_post_js(gemma_cpp_js ${CMAKE_CURRENT_SOURCE_DIR}/src/post.js)
target_link_options(gemma_cpp_js PRIVATE
  -sEXPORT_ES6=1
  -sMODULARIZE=1
  -lembind
  -sALLOW_MEMORY_GROWTH=1
  -sMAXIMUM_MEMORY=16GB
  -sINITIAL_MEMORY=2GB
  -sSTACK_SIZE=2MB
  -sPTHREAD_POOL_SIZE_STRICT=0
)

# --- Apply the linker flag if the string is not empty ---
if(ENVIRONMENT)
  target_link_options(gemma_cpp_js PRIVATE "-sENVIRONMENT=${ENVIRONMENT}")
endif()

set_target_properties(gemma_cpp_js PROPERTIES SUFFIX ".mjs")

target_link_libraries(gemma_cpp_js libgemma)

FetchContent_GetProperties(gemma)
target_include_directories(gemma_cpp_js PRIVATE ${gemma_SOURCE_DIR})
