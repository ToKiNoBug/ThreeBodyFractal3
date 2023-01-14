project(ThreeBodyFractal3_libthreebodyfractal LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

enable_language(CUDA)

# set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)
find_package(PNG REQUIRED)

include(${CMAKE_SOURCE_DIR}/cmake/configure_fractal_utils.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/configure_libnbtplusplus.cmake)

# include(${CMAKE_SOURCE_DIR}/cmake/configure_cpp_nbt.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/configure_nlohmann_json.cmake)

find_package(
  fractal_utils ${version_fu}
  COMPONENTS core_utils png_utils
  REQUIRED)

find_package(libNBT++ ${version_libnbtpp} REQUIRED)

find_package(LibLZMA REQUIRED)

add_library(
  threebodyfractal STATIC
  libthreebodyfractal.h
  libthreebodyfractal.cpp
  save_binfile.cpp
  load_binfile.cpp
  memory_resource.h
  memory_resource.cpp
  coloring.cpp
  save_load_json.cpp
)

target_include_directories(threebodyfractal PUBLIC ${fractal_utils_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(
  threebodyfractal PRIVATE ${LIBLZMA_INCLUDE_DIRS} ${nlohmann_json_include_dir})

target_compile_features(threebodyfractal PUBLIC cxx_std_20)

# target_compile_features(${target} PRIVATE cxx_std_23)
target_link_libraries(
  threebodyfractal
  PUBLIC OpenMP::OpenMP_CXX
  threebody
  PNG::PNG
  fractal_utils::core_utils
  fractal_utils::png_utils
  cudathreebody
  LibLZMA::LibLZMA
  libNBT++::NBT++)

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  OR(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"))
  target_compile_options(threebodyfractal PRIVATE -mavx -mavx2 -mfma)
endif()

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  OR(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
  target_compile_options(threebodyfractal PRIVATE /arch:AVX /arch:AVX2)
  target_compile_options(threebodyfractal PUBLIC /wd4819 /wd4068)
endif()

# find_package(Boost COMPONENTS container) if(${BOOST_FOUND}) message(STATUS
# "Boost include dirs = " ${Boost_INCLUDE_DIRS})
# target_include_directories(threebodyfractal PRIVATE ${Boost_INCLUDE_DIRS})
# else() message(STATUS "Boost::container not found. Use std::map instead")
# target_compile_definitions(threebodyfractal PRIVATE "-DNBT_MAP_TYPE=std::map")
# endif()
if(NOT ${tb_build_examples})
  return()
endif()

include(${CMAKE_SOURCE_DIR}/cmake/get_lto_options.cmake)

add_executable(test_tbf test_tbf.cpp)
target_link_libraries(test_tbf PRIVATE threebodyfractal)
target_compile_options(test_tbf PRIVATE ${ThreebodyFractal_lto_options})

add_executable(test_tbf_load test_tbf_load.cpp)
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

add_executable(test_render test_render.cpp)
target_link_libraries(test_render PRIVATE threebodyfractal)
target_compile_options(test_render PRIVATE ${ThreebodyFractal_lto_options})

add_executable(test_tbf_load_2 test_tbf_load_2.cpp)
target_link_libraries(test_tbf_load_2 PRIVATE threebodyfractal)
target_compile_options(test_tbf_load_2 PRIVATE ${ThreebodyFractal_lto_options})
