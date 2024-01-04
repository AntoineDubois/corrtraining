cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(corrtraining)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wfatal-errors")
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Add the path to the library header files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/dynet)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen) # eigen is header-only

# Add the path to the library binary
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/dynet/build/dynet)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory("include")

add_executable(corrtraining main.cpp)
target_link_libraries(corrtraining PUBLIC dynet Eigen3::Eigen include)
target_include_directories(corrtraining PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_SOURCE_DIR}/include"
                           )

# Set up the testing library
add_subdirectory("tests")
enable_testing()

target_link_libraries(test_utils PUBLIC dynet Eigen3::Eigen include)
target_include_directories(test_utils PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_SOURCE_DIR}/include"
                           )
add_test(NAME test_utils COMMAND $<TARGET_FILE:test_utils> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_action PUBLIC dynet Eigen3::Eigen include)
target_include_directories(test_action PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_SOURCE_DIR}/include"
                           )
add_test(NAME test_action COMMAND $<TARGET_FILE:test_action> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_state PUBLIC dynet Eigen3::Eigen include)
target_include_directories(test_state PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_SOURCE_DIR}/include"
                           )
add_test(NAME test_state COMMAND $<TARGET_FILE:test_state> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_reward PUBLIC dynet Eigen3::Eigen include)
target_include_directories(test_reward PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_SOURCE_DIR}/include"
                           )
add_test(NAME test_reward COMMAND $<TARGET_FILE:test_reward> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_buffer PUBLIC dynet Eigen3::Eigen include)
target_include_directories(test_buffer PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_SOURCE_DIR}/include"
                           )
add_test(NAME test_buffer COMMAND $<TARGET_FILE:test_buffer> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_agent PUBLIC dynet Eigen3::Eigen include)
target_include_directories(test_agent PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_SOURCE_DIR}/include"
                           )
add_test(NAME test_agent COMMAND $<TARGET_FILE:test_agent> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_model PUBLIC dynet Eigen3::Eigen include)
target_include_directories(test_model PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_SOURCE_DIR}/include"
                           )
add_test(NAME test_model COMMAND $<TARGET_FILE:test_model> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_environment PUBLIC dynet Eigen3::Eigen include)
target_include_directories(test_environment PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_SOURCE_DIR}/include"
                           )
add_test(NAME test_environment COMMAND $<TARGET_FILE:test_environment> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_actor_critic PUBLIC dynet Eigen3::Eigen include)
target_include_directories(test_actor_critic PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_SOURCE_DIR}/include"
                           )
add_test(NAME test_actor_critic COMMAND $<TARGET_FILE:test_actor_critic> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})