cmake_minimum_required(VERSION 3.5)


#find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

#message(STATUS ${Eigen3_INCLUDE_DIR})

add_executable(ThreeBodyFractal
    main.cpp
    Derivative.h
    MultiRec.h
    Solvers.h
    )

if(EXISTS ${CMAKE_SOURCE_DIR}/compile_commands.json)
    file(REMOVE ${CMAKE_SOURCE_DIR}/compile_commands.json)
endif()

target_link_libraries(ThreeBodyFractal PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(ThreeBodyFractal PRIVATE -mavx -mavx2)