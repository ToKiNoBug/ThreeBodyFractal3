project(ThreeBodyFractal3_libcudathreebody LANGUAGES CXX)
cmake_minimum_required(VERSION 3.15)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

enable_language(CUDA)

# find_package(Eigen3 REQUIRED)

# include_directories(${CMAKE_SOURCE_DIR}/libthreebody)
add_library(cudathreebody STATIC libcudathreebody.h libcudathreebody.cu
                                 internal.h simulate.cu)
target_link_libraries(cudathreebody PUBLIC threebody)

if(NOT ${tb_build_examples})
  return()
endif()

find_package(OpenMP REQUIRED)

add_executable(test_cutb test_cutb.cpp)
target_link_libraries(test_cutb PRIVATE cudathreebody threebody
                                        OpenMP::OpenMP_CXX)

# target_include_directories(cudathreebody PUBLIC
# ${CMAKE_SOURCE_DIR}/libthreebody)
