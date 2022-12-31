project(ThreeBodyFractal3_libthreebodyfractal LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenMP REQUIRED)

include(${CMAKE_SOURCE_DIR}/cmake/configure_fractal_utils.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/configure_cpp_nbt.cmake)

find_package(PNG REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "/media/jeremiah/D/Git/build-ThreeBodyFractal3-linux/3rdParty/FractalUtils/install")

find_package(fractal_utils ${version_fu} COMPONENTS core_utils png_utils REQUIRED)

find_package(LibLZMA REQUIRED)

add_library(threebodyfractal STATIC
    threebodyfractal.h
    threebodyfractal.cpp
    save_binfile.cpp
    load_binfile.cpp)

include_directories(${fractal_utils_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/libthreebody
    ${LIBLZMA_INCLUDE_DIRS}
    ${cpp_nbt_include_dir}
)

set(tbf_dependents
    threebody
    PNG::PNG
    fractal_utils::core_utils
    fractal_utils::png_utils
    OpenMP::OpenMP_CXX
    LibLZMA::LibLZMA)

target_link_libraries(threebodyfractal PUBLIC

    OpenMP::OpenMP_CXX
    threebody
    PNG::PNG
    fractal_utils::core_utils
    fractal_utils::png_utils)

find_package(Boost COMPONENTS container)

if(${BOOST_FOUND})
    message(STATUS "Boost include dirs = " ${Boost_INCLUDE_DIRS})
    target_include_directories(threebodyfractal PRIVATE ${Boost_INCLUDE_DIRS})
else()
    message(STATUS "Boost::container not found. Use std::map instead")
    target_compile_definitions(threebodyfractal PRIVATE "-DNBT_MAP_TYPE=std::map")
endif()

include(${CMAKE_SOURCE_DIR}/cmake/get_lto_options.cmake)

add_executable(test_tbf test_tbf.cpp)
target_link_libraries(test_tbf PUBLIC threebodyfractal ${tbf_dependents})
target_compile_options(test_tbf PRIVATE ${ThreebodyFractal_lto_options})

add_executable(test_tbf_load test_tbf2.cpp)
target_link_libraries(test_tbf_load PUBLIC threebodyfractal ${tbf_dependents})
target_compile_options(test_tbf_load PRIVATE ${ThreebodyFractal_lto_options})

add_executable(test_tbf_other_input test_tbf_other_input.cpp)
target_link_libraries(test_tbf_other_input PUBLIC threebodyfractal ${tbf_dependents})
target_compile_options(test_tbf_other_input PRIVATE ${ThreebodyFractal_lto_options})