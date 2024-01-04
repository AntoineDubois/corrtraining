The project Deep-Correlation uses deep reinforcement learning to discover a an algorithm that maximises the sum of determinant of a covariance matrix’s diagonal blocks.  


# Building the package
Tutorial at: https://dynet.readthedocs.io/en/latest/install.html

1. mkdir third_party
2. cd third_party
3. Download the latest version of Eigen: curl -O https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
4. Decompress the latest version: tar -xvf eigen-3.3.9.tar
4. git clone https://github.com/clab/dynet.git
5. cd dynet
6. mkdir build
7. cd build
8. cmake .. -DEIGEN3_INCLUDE_DIR=path/to/eigen -DENABLE_CPP_EXAMPLES=ON
9. make -j 2
10. ./examples/train_xor /* check dynet built correctly */

11. create CMakeLists.txt to link dynet and my project
    11.a include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/dynet) /* Add the path to the library header files */
    11.b link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/dynet/build/dynet) /* Add the path to the library binary */
    11.c target_link_libraries(corrtraining PUBLIC dynet)

