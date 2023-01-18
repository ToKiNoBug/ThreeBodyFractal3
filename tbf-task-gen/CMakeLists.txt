include(${CMAKE_SOURCE_DIR}/cmake/configure_cli11.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/configure_nlohmann_json.cmake)
enable_language(CUDA)

add_library(tbf-task STATIC tbf-task.cpp)
target_include_directories(tbf-task PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(tbf-task PRIVATE ${nlohmann_json_include_dir})
target_compile_features(tbf-task PUBLIC cxx_std_20)

add_executable(tbf-task-gen tbf-task-gen.cpp)
target_link_libraries(tbf-task-gen PRIVATE tbf-task)

target_include_directories(tbf-task-gen PRIVATE ${cli11_include_dir})
target_link_libraries(tbf-task-gen PRIVATE threebodyfractal)
