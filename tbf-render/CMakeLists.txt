include(${CMAKE_SOURCE_DIR}/cmake/configure_cli11.cmake)

enable_language(CUDA)

add_executable(tbf-render tbf-render.cpp)

target_link_libraries(tbf-render PRIVATE tbf-task threebodyfractal)

target_include_directories(tbf-render PRIVATE ${cli11_include_dir})