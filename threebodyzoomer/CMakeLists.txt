project(ThreeBodyFractal3_libthreebodyfractal LANGUAGES CXX)
cmake_minimum_required(VERSION 3.15)

enable_language(CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(QT NAMES Qt6 COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets)

if(NOT ${QT_FOUND})
  message(
    WARNING "Qt6 Widgets not found. threebodyfractalzoomer will not be built.")
  return()
endif()

find_package(fractal_utils ${version_fu} COMPONENTS core_utils png_utils
                                                    zoom_utils)

if(NOT ${fractal_utils_FOUND})
  message(
    WARNING
      "fractal_utils::zoom_utils not found. threebodyfractalzoomer will not be built."
  )
  return()
endif()

find_package(OpenMP REQUIRED)

qt_add_executable(threebodyzoomer MANUAL_FINALIZATION zoomer.cpp)

target_include_directories(
  threebodyzoomer PRIVATE ${CMAKE_SOURCE_DIR}/libthreebody
                          ${CMAKE_SOURCE_DIR}/libthreebodyfractal)

target_link_libraries(
  threebodyzoomer
  PRIVATE Qt${QT_VERSION_MAJOR}::Widgets fractal_utils::core_utils
          OpenMP::OpenMP_CXX fractal_utils::zoom_utils threebodyfractal)

qt_finalize_executable(threebodyzoomer)
