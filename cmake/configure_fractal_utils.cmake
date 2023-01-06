# list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}"/3rdParty/install")

# list(APPEND CMAKE_PREFIX_PATH "D:/Git/build-FractalUtils-win/install")

set(version_fu 1.1.1)

find_package(fractal_utils ${version_fu} COMPONENTS core_utils png_utils)

if(${fractal_utils_FOUND})
  message(STATUS "fractal_utils found.")
  return()
endif()

message(STATUS "fractal_utils not found. Download and reinstall.")

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/3rdParty/FractalUtils/.git)
  message(STATUS "Downloading fractal_utils ...")

  execute_process(
    COMMAND git clone "https://github.com/ToKiNoBug/FractalUtils.git"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdParty COMMAND_ERROR_IS_FATAL ANY)
endif()

execute_process(
  COMMAND git checkout "v${version_fu}"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdParty/FractalUtils
                    COMMAND_ERROR_IS_FATAL ANY
  OUTPUT_QUIET)

message(STATUS "CMake is configuring fractal_utils ...")

message(STATUS "CMAKE_GENERATOR = " ${CMAKE_GENERATOR})
message(STATUS "CMAKE_MAKE_PROGRAM = " ${CMAKE_MAKE_PROGRAM})
message(STATUS "CMAKE_RC_COMPILER = " ${CMAKE_RC_COMPILER})

# return() set(command_temp "cmake -G ${CMAKE_GENERATOR} -S
# ${CMAKE_SOURCE_DIR}/3rdParty/FractalUtils -B
# ${CMAKE_BINARY_DIR}/3rdParty/FractalUtils/build
# -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
# -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
# -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
# -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}/3rdParty/FractalUtils/install
# -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")

# message(STATUS "command is : " ${command_temp})
set(fu_build_dir ${CMAKE_BINARY_DIR}/3rdParty/FractalUtils/build)

set(fu_install_dir ${CMAKE_BINARY_DIR}/3rdParty/FractalUtils/install)

set(command_args
    -G
    ${CMAKE_GENERATOR}
    -S
    "${CMAKE_SOURCE_DIR}/3rdParty/FractalUtils"
    -B
    "${fu_build_dir}"
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_INSTALL_PREFIX:PATH=${fu_install_dir}"
    "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")

if((CMAKE_CXX_COMPILER_ID STREQUAL MSVC) OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT
                                             STREQUAL MSVC))
  if(DEFINED CMAKE_RC_COMPILER)
    set(command_args ${command_args}
                     "-DCMAKE_RC_COMPILER:FILEPATH=${CMAKE_RC_COMPILER}")
  endif()
endif()

message(STATUS "The command is : cmake" ${command_args})

execute_process(COMMAND cmake ${command_args} COMMAND_ERROR_IS_FATAL ANY)

message(STATUS "CMake is building 3rdParty project FractalUtils...")

execute_process(
  COMMAND cmake --build . --parallel
  WORKING_DIRECTORY ${fu_build_dir} COMMAND_ERROR_IS_FATAL ANY
  OUTPUT_VARIABLE output_temp)

message(STATUS "CMake is installing 3rdParty project FractalUtils...")

execute_process(
  COMMAND cmake --install .
  WORKING_DIRECTORY ${fu_build_dir} COMMAND_ERROR_IS_FATAL ANY
  OUTPUT_VARIABLE output_temp)

set(fractal_utils_INCLUDE_DIR ${fu_install_dir}/include)

set(CFP_temp CMAKE_PREFIX_PATH)

list(APPEND CMAKE_PREFIX_PATH ${fu_install_dir})

find_package(fractal_utils ${version_fu} COMPONENTS core_utils png_utils)

if(NOT ${fractal_utils_FOUND})
  set(CMAKE_PREFIX_PATH ${CFP_temp})
  message(
    FATAL_ERROR
      "Unprecedented error : fractal_utils installed but find_package failed to find it. Adding its installation dir failed to fix this problem."
  )
endif()

unset(CFP_temp)

message("fractal_utils_INCLUDE_DIR = " ${fractal_utils_INCLUDE_DIR})
message(
  "FractalUtils installed and is avaliable by find_package. CMAKE_PREFIX_PATH = "
  ${CMAKE_PREFIX_PATH})
