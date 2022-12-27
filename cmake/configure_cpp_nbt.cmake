# https://github.com/SpockBotMC/cpp-nbt

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/3rdParty/cpp-nbt/nbt.hpp)
    message(STATUS "cpp-nbt not found. Downloading...")

    execute_process(COMMAND git clone https://github.com/ToKiNoBug/cpp-nbt
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdParty
        COMMAND_ERROR_IS_FATAL ANY)
else()
    message(STATUS "cpp-nbt found.")
endif()

execute_process(COMMAND git checkout code-from-buffer
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdParty/cpp-nbt
    COMMAND_ERROR_IS_FATAL ANY OUTPUT_QUIET)

set(cpp_nbt_include_dir ${CMAKE_SOURCE_DIR}/3rdParty/cpp-nbt)