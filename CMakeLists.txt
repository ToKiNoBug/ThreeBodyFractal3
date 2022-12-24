project(ThreeBodyFractal3 LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(NOT DEFINED threebodyfractal_eigen_include_dir)
        set(threebodyfractal_eigen_include_dir "D:/CppLibs/eigen-3.4.0")
    endif()

    include_directories(${threebodyfractal_eigen_include_dir})
else()
    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

if(NOT DEFINED CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH)
endif()

# add_compile_definitions("_USE_MATH_DEFINES")
add_subdirectory(libthreebody)
add_subdirectory(libthreebodyfractal)

# add_subdirectory(old)