project(ThreeBodyFractal3_libcudathreebody LANGUAGES CXX)
cmake_minimum_required(VERSION 3.15)

enable_language(CUDA)

# find_package(Eigen3 REQUIRED)

# include_directories(${CMAKE_SOURCE_DIR}/libthreebody)

add_library(cudathreebody STATIC libcudathreebody.h libcudathreebody.cu
                                 internal.h simulate.cu)

# target_include_directories(cudathreebody PUBLIC
# ${CMAKE_SOURCE_DIR}/libthreebody)
