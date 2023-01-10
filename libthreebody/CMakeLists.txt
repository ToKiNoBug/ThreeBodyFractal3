project(ThreeBodyFractal3_libthreebody LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5)

# find_package(Eigen3 REQUIRED) find_package(OpenMP REQUIRED)
add_library(threebody STATIC libthreebody.h libthreebody.cpp)

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
   OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"))
  target_compile_options(threebody PRIVATE -mavx -mavx2 -mfma)
endif()

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
   OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
  target_compile_options(threebody PRIVATE /arch:AVX /arch:AVX2
                                           /std:c++${CMAKE_CXX_STANDARD})
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
     OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"))
    target_compile_options(threebody PRIVATE -Ofast)
  endif()

  if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
     OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
    target_compile_options(threebody PRIVATE /Ox /Ot)
  endif()
endif()

if(NOT ${tb_build_examples})
  return()
endif()

add_executable(test_libthreebody test_tb.cpp)
target_link_libraries(test_libthreebody PRIVATE threebody)

add_executable(test_loadD3B3 test_loadD3B3.cpp)
target_link_libraries(test_loadD3B3 PRIVATE threebody)
