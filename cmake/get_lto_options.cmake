unset(ThreebodyFractal_lto_options)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(ThreebodyFractal_lto_options "-flto")
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(ThreebodyFractal_lto_options "-flto")
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    set(ThreebodyFractal_lto_options)
endif()