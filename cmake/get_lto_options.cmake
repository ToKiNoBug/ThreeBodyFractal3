unset(ThreebodyFractal_lto_options)


if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
   OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"))
    set(ThreebodyFractal_lto_options "-flto")
endif()

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
    set(ThreebodyFractal_lto_options)
endif()