project(ThreeBodyFractal3_libthreebodyfractal LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

include(${CMAKE_SOURCE_DIR}/cmake/configure_fractal_utils.cmake)

find_package(PNG REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "/media/jeremiah/D/Git/build-ThreeBodyFractal3-linux/3rdParty/FractalUtils/install")

find_package(fractal_utils ${version_fu} COMPONENTS core_utils png_utils REQUIRED)

find_package(LibLZMA REQUIRED)

add_library(threebodyfractal STATIC
    threebodyfractal.h
    threebodyfractal.cpp
    save_binfile.cpp
    load_binfile.cpp)

include_directories(${fractal_utils_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/libthreebody ${LIBLZMA_INCLUDE_DIRS})

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

add_executable(test_tbf test_tbf.cpp)

target_link_libraries(test_tbf PUBLIC threebodyfractal ${tbf_dependents})