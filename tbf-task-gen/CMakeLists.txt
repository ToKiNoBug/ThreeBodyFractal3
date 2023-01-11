include(${CMAKE_SOURCE_DIR}/cmake/configure_cli11.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/configure_nlohmann_json.cmake)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(tbf-task-gen tbf-task-gen.cpp)

target_include_directories(tbf-task-gen PRIVATE ${cli11_include_dir}
                                                ${nlohmann_json_include_dir})
target_link_libraries(tbf-task-gen PRIVATE threebodyfractal)
