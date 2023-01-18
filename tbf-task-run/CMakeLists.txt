project(ThreeBodyFractal3_libthreebodyfractal LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5)

enable_language(CUDA)

include(${CMAKE_SOURCE_DIR}/cmake/configure_cli11.cmake)

add_executable(tbf-task-run tbf-task-run.cpp)
target_link_libraries(tbf-task-run PRIVATE tbf-task threebodyfractal)
target_include_directories(tbf-task-run PRIVATE ${cli11_include_dir})