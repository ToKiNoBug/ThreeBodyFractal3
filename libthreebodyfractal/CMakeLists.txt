project(ThreeBodyFractal3_libthreebodyfractal LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 23)
find_package(OpenMP REQUIRED)

include(${CMAKE_SOURCE_DIR}/cmake/configure_fractal_utils.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/configure_cpp_nbt.cmake)

find_package(PNG REQUIRED)

# list( APPEND CMAKE_PREFIX_PATH
# "/media/jeremiah/D/Git/build-ThreeBodyFractal3-linux/3rdParty/FractalUtils/install"
# )
find_package(
  fractal_utils ${version_fu}
  COMPONENTS core_utils png_utils
  REQUIRED)

find_package(LibLZMA REQUIRED)

add_library(
  threebodyfractal STATIC
  libthreebodyfractal.h
  libthreebodyfractal.cpp
  save_binfile.cpp
  load_binfile.cpp
  memory_resource.h
  memory_resource.cpp
  coloring.cpp)

target_include_directories(
  threebodyfractal
  PUBLIC ${CMAKE_SOURCE_DIR}/libthreebody ${fractal_utils_INCLUDE_DIR}
         ${LIBLZMA_INCLUDE_DIRS} ${cpp_nbt_include_dir})

target_link_libraries(
  threebodyfractal
  PUBLIC OpenMP::OpenMP_CXX
         threebody
         PNG::PNG
         fractal_utils::core_utils
         fractal_utils::png_utils
         cudathreebody
         LibLZMA::LibLZMA)

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
   OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"))
  target_compile_options(threebodyfractal PRIVATE -mavx -mavx2 -mfma)
endif()

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
   OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
  target_compile_options(threebodyfractal PRIVATE /arch:AVX /arch:AVX2
                                                  /std:c++${CMAKE_CXX_STANDARD})
endif()

find_package(Boost COMPONENTS container)

if(${BOOST_FOUND})
  message(STATUS "Boost include dirs = " ${Boost_INCLUDE_DIRS})
  target_include_directories(threebodyfractal PRIVATE ${Boost_INCLUDE_DIRS})
else()
  message(STATUS "Boost::container not found. Use std::map instead")
  target_compile_definitions(threebodyfractal PRIVATE "-DNBT_MAP_TYPE=std::map")
endif()

if(NOT ${tb_build_examples})
  return()
endif()

include(${CMAKE_SOURCE_DIR}/cmake/get_lto_options.cmake)

add_executable(test_tbf test_tbf.cpp)
target_link_libraries(test_tbf PRIVATE threebodyfractal)
target_compile_options(test_tbf PRIVATE ${ThreebodyFractal_lto_options})

add_executable(test_tbf_load test_tbf2.cpp)
target_link_libraries(test_tbf_load PRIVATE threebodyfractal)
target_compile_options(test_tbf_load PRIVATE ${ThreebodyFractal_lto_options})

add_executable(test_tbf_other_input test_tbf_other_input.cpp)
target_link_libraries(test_tbf_other_input PRIVATE threebodyfractal)
target_compile_options(test_tbf_other_input
                       PRIVATE ${ThreebodyFractal_lto_options})

add_executable(test_tbf_allocator test_allocator.cpp)
target_link_libraries(test_tbf_allocator PRIVATE threebodyfractal)
target_compile_options(test_tbf_allocator
                       PRIVATE ${ThreebodyFractal_lto_options})
