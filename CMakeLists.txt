project(ThreeBodyFractal3 LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5)

include_directories("D:/CppLibs/eigen-3.4.0")

#find_package(Eigen3 REQUIRED)


#add_compile_definitions("_USE_MATH_DEFINES")

add_subdirectory(libthreebody)
#add_subdirectory(old)