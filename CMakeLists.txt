project(ThreeBodyFractal3 LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5)

if(EXISTS ${CMAKE_SOURCE_DIR}/compile_commands.json)
  message(STATUS removed)
  file(REMOVE ${CMAKE_SOURCE_DIR}/compile_commands.json)
endif()

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  if(NOT DEFINED threebodyfractal_eigen_include_dir)
    set(threebodyfractal_eigen_include_dir "D:/CppLibs/eigen-3.4.0")
  endif()

  set(EIGEN3_INCLUDE_DIR ${threebodyfractal_eigen_include_dir})

# include_directories(${threebodyfractal_eigen_include_dir})
else()
  find_package(Eigen3 REQUIRED)

  # include_directories(${EIGEN3_INCLUDE_DIR})
endif()

if(NOT DEFINED CMAKE_PREFIX_PATH)
  set(CMAKE_PREFIX_PATH)
endif()

include(cmake/add_compiler_path_to_prefix.cmake)

if(NOT DEFINED tb_build_examples)
  set(tb_build_examples ON)
endif()

# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -pg)
# add_compile_definitions("_USE_MATH_DEFINES")
add_subdirectory(libthreebody)
add_subdirectory(libcudathreebody)
add_subdirectory(libthreebodyfractal)
add_subdirectory(threebodyzoomer)
add_subdirectory(tbfsingle)
add_subdirectory(tbfcvt)
add_subdirectory(tbf-task-gen)
add_subdirectory(tbf-task-run)

# add_subdirectory(old)
