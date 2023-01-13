project(ThreeBodyFractal3_libcudathreebody LANGUAGES CXX)
cmake_minimum_required(VERSION 3.15)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

enable_language(CUDA)

# set(CMAKE_CXX_FLAGS /w4819 /w4068)
add_library(cudathreebody STATIC libcudathreebody.h libcudathreebody.cu
                                 internal.h simulate.cu)
target_include_directories(cudathreebody PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cudathreebody PUBLIC threebody)

target_compile_features(cudathreebody PRIVATE cxx_std_17)

target_compile_options(cudathreebody BEFORE
                       PRIVATE --diag-suppress 20012,20013,20015,20054,20236)

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
   OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
  target_compile_options(cudathreebody PRIVATE "-Xcompiler=/wd4819"
                                               "-Xcompiler=/wd4068")
endif()

if(NOT ${tb_build_examples})
  return()
endif()

find_package(OpenMP REQUIRED)

add_executable(test_cutb test_cutb.cpp)
target_link_libraries(test_cutb PUBLIC cudathreebody threebody
                                       OpenMP::OpenMP_CXX)
target_compile_features(test_cutb PRIVATE cxx_std_20)
