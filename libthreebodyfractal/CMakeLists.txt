project(ThreeBodyFractal3_libthreebodyfractal LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

include(${CMAKE_SOURCE_DIR}/cmake/configure_fractal_utils.cmake)

find_package(PNG REQUIRED)
find_package(fractal_utils 1.0.3 COMPONENTS core_utils png_utils REQUIRED)

add_library(threebodyfractal STATIC
    threebodyfractal.h
    threebodyfractal.cpp)

target_include_directories(threebodyfractal PUBLIC ${fractal_utils_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/libthreebody)
target_link_libraries(threebodyfractal PUBLIC
    threebody
    PNG::PNG
    fractal_utils::core_utils
    fractal_utils::png_utils
    OpenMP::OpenMP_CXX)