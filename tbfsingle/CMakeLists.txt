include(${CMAKE_SOURCE_DIR}/cmake/configure_cli11.cmake)

enable_language(CUDA)

add_executable(tbfsingle tbfsingle.cpp run_compute.cpp run_render.cpp)
target_link_libraries(tbfsingle PRIVATE threebodyfractal)
target_include_directories(
  tbfsingle PRIVATE ${cli11_include_dir}
                    ${CMAKE_SOURCE_DIR}/libthreebodyfractal)
