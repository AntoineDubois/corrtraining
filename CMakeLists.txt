cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(corrtraining)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the path to the library header files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/dynet)
# Add the path to the library binary
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/dynet/build/dynet)

add_executable(corrtraining main.cpp)
target_link_libraries(corrtraining PUBLIC dynet)
target_include_directories(corrtraining PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_SOURCE_DIR}"
                           )