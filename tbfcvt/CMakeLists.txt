include(${CMAKE_SOURCE_DIR}/cmake/configure_cli11.cmake)

enable_language(CUDA)

add_executable(tbfcvt tbfcvt.cpp)
target_link_libraries(tbfcvt PRIVATE threebodyfractal)
target_include_directories(
    tbfcvt PRIVATE ${cli11_include_dir})