set(version_libnbtpp 2.5.1)

find_package(libNBT++ ${version_libnbtpp})

if(${libNBT++_FOUND})
  message(STATUS "libNBT++ Found.")
  return()
endif()

message(STATUS "libNBT++ not found. Download and reinstall.")

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/3rdParty/libnbtplusplus/.git)
  message(STATUS "Downloading libNBT++ ...")

  execute_process(
    COMMAND git clone "https://github.com/ToKiNoBug/libnbtplusplus.git"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdParty COMMAND_ERROR_IS_FATAL ANY)
endif()

# execute_process( COMMAND git checkout "v${version_libnbtpp}" WORKING_DIRECTORY
# ${CMAKE_SOURCE_DIR}/3rdParty/libnbtplusplus COMMAND_ERROR_IS_FATAL ANY
# OUTPUT_QUIET)

message(STATUS "CMake is configuring libNBT++(aka libnbtplusplus) ...")

# message(STATUS "CMAKE_GENERATOR = " ${CMAKE_GENERATOR})

# message(STATUS "CMAKE_MAKE_PROGRAM = " ${CMAKE_MAKE_PROGRAM})

# message(STATUS "CMAKE_RC_COMPILER = " ${CMAKE_RC_COMPILER})

set(libnbtpp_build_dir ${CMAKE_BINARY_DIR}/3rdParty/libnbtplusplus/build)

set(libnbtpp_install_dir ${CMAKE_BINARY_DIR}/3rdParty/libnbtplusplus/install)

set(command_args
    -G
    ${CMAKE_GENERATOR}
    -S
    ${CMAKE_SOURCE_DIR}/3rdParty/libnbtplusplus
    -B
    ${libnbtpp_build_dir}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_INSTALL_PREFIX:PATH=${libnbtpp_install_dir}
    -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
    -DNBT_BUILD_TESTS:BOOL=FALSE
    -DNBT_BUILD_SHARED:BOOL=FALSE
    -DNBT_USE_ZLIB:BOOL=TRUE)

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
   OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
  if(DEFINED CMAKE_RC_COMPILER)
    set(command_args ${command_args}
                     "-DCMAKE_RC_COMPILER:FILEPATH=${CMAKE_RC_COMPILER}")
  endif()
endif()

message(STATUS "The command is : cmake" ${command_args})

execute_process(COMMAND cmake ${command_args} COMMAND_ERROR_IS_FATAL ANY)

message(
  STATUS "CMake is building 3rdParty project libNBT++(aka libnbtplusplus)...")

execute_process(
  COMMAND cmake --build . --parallel
  WORKING_DIRECTORY ${libnbtpp_build_dir} COMMAND_ERROR_IS_FATAL ANY
  OUTPUT_VARIABLE output_temp)

message(
  STATUS "CMake is installing 3rdParty project libNBT++(aka libnbtplusplus)...")

execute_process(
  COMMAND cmake --install .
  WORKING_DIRECTORY ${libnbtpp_build_dir} COMMAND_ERROR_IS_FATAL ANY
  OUTPUT_VARIABLE output_temp)

# set(libnbtplusplus_utils_INCLUDE_DIR ${libnbtpp_install_dir}/include)

set(CFP_temp CMAKE_PREFIX_PATH)

list(APPEND CMAKE_PREFIX_PATH ${libnbtpp_install_dir})

find_package(libNBT++ ${version_libnbtpp})

if(NOT ${libNBT++_FOUND})
  set(CMAKE_PREFIX_PATH ${CFP_temp})
  message(
    FATAL_ERROR
      "Unprecedented error : libNBT++(aka libnbtplusplus) installed but find_package failed to find it. Adding its installation dir failed to fix this problem."
  )
endif()

unset(CFP_temp)

# message("libnbtplusplus_utils_INCLUDE_DIR = "
# ${libnbtplusplus_utils_INCLUDE_DIR})
message(
  "libNBT++(aka libnbtplusplus) installed and is avaliable by find_package. CMAKE_PREFIX_PATH = "
  ${CMAKE_PREFIX_PATH})
