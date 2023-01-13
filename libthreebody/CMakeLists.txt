project(ThreeBodyFractal3_libthreebody LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5)

# find_package(Eigen3 REQUIRED) find_package(OpenMP REQUIRED)
add_library(threebody STATIC libthreebody.h libthreebody.cpp)

target_include_directories(threebody PUBLIC ${EIGEN3_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(threebody PRIVATE cxx_std_20)

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
   OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"))
  target_compile_options(threebody PRIVATE -mavx -mavx2 -mfma)
endif()

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
   OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
  target_compile_options(threebody PRIVATE /arch:AVX /arch:AVX2
                                           /std:c++${CMAKE_CXX_STANDARD})
endif()

# Disable some warnings
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  target_compile_definitions(threebody PUBLIC _CRT_SECURE_NO_WARNINGS)
endif()

# Add optimization options
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
     OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"))
    target_compile_options(threebody PRIVATE -Ofast)
    set(disable_warnings)
  endif()

  if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
     OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
    target_compile_options(threebody PRIVATE /Ox /Ot)
    set(disable_warnings /wd4819)
  endif()
endif()

# Disable stupid warnings
target_compile_options(threebody PRIVATE ${disable_warnings})

if(NOT ${tb_build_examples})
  return()
endif()

add_executable(test_libthreebody test_tb.cpp)
target_link_libraries(test_libthreebody PRIVATE threebody)
target_compile_features(test_libthreebody PRIVATE cxx_std_20)

add_executable(test_loadD3B3 test_loadD3B3.cpp)
target_link_libraries(test_loadD3B3 PRIVATE threebody)
target_compile_features(test_loadD3B3 PRIVATE cxx_std_20)
