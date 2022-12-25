project(ThreeBodyFractal3_libthreebody LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

#find_package(Eigen3 REQUIRED)
#find_package(OpenMP REQUIRED)

add_library(threebody STATIC 
    libthreebody.h 
    libthreebody.cpp)

target_compile_options(threebody PUBLIC -mavx -mavx2 -mfma)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
target_compile_options(threebody PRIVATE -Ofast)
endif()

add_executable(test_libthreebody test_tb.cpp)

target_link_libraries(test_libthreebody PRIVATE threebody)