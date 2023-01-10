# https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp

cmake_minimum_required(VERSION 3.19)

if(DEFINED nlohmann_json_include_dir)
  if(EXISTS ${nlohmann_json_include_dir}/nlohmann/json.hpp)
    message(STATUS "nlohmann/json.hpp found at "
                   ${nlohmann_json_include_dir}/nlohmann/json.hpp)
    return()
  else()
    message(WARNING "Assigned nlohmann_json_include_dir to be "
                    ${nlohmann_json_include_dir}
                    " but failed to find nlohmann/json.hpp")
    unset(nlohmann_json_include_dir)
  endif()
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/3rdParty/nlohmann/nlohmann/json.hpp)
  message(STATUS "nlohmann/json.hpp found at "
                 ${CMAKE_SOURCE_DIR}/3rdParty/nlohmann/nlohmann/json.hpp)
  set(nlohmann_json_include_dir ${CMAKE_SOURCE_DIR}/3rdParty/nlohmann)
  return()
endif()

message(STATUS "Downloading nlohmann/json.hpp ......")

file(DOWNLOAD
     https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
     ${CMAKE_SOURCE_DIR}/3rdParty/nlohmann/nlohmann/json.hpp)

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/3rdParty/nlohmann/nlohmann/json.hpp)
  message(ERROR "Failed to download nlohmann/json.")
  return()
endif()

message(STATUS "nlohmann/json downloaded successfully.")
set(nlohmann_json_include_dir ${CMAKE_SOURCE_DIR}/3rdParty/nlohmann)
