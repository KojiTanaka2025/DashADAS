cmake_minimum_required(VERSION 3.16)
project(dashadas_perception VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(DASHADAS_BUILD_EXAMPLES "Build example clients" ON)
option(DASHADAS_DOWNLOAD_STB "Download stb_image.h for JPEG/PNG CLI input" ON)

set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}" CACHE PATH "Qualcomm QAIRT / QNN SDK root")
if(NOT QNN_SDK_ROOT)
  message(FATAL_ERROR "QNN_SDK_ROOT is required (CMake cache or environment)")
endif()
if(NOT EXISTS "${QNN_SDK_ROOT}/include/QNN/QnnInterface.h")
  message(FATAL_ERROR "QNN headers not found under ${QNN_SDK_ROOT}/include/QNN")
endif()

add_library(dashadas_perception SHARED
  src/api.cpp
  src/letterbox.cpp
  src/yolo_decode.cpp
  src/qnn_backend.cpp
)

set_target_properties(dashadas_perception PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 0
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  OUTPUT_NAME dashadas_perception
)

target_compile_definitions(dashadas_perception PRIVATE DASHADAS_BUILD)
target_compile_options(dashadas_perception PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(dashadas_perception
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${QNN_SDK_ROOT}/include/QNN
)
target_link_libraries(dashadas_perception PRIVATE ${CMAKE_DL_LIBS} pthread)

include(GNUInstallDirs)
install(TARGETS dashadas_perception
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/dashadas DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/dashadas_perception.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/dashadas_perception.pc
  @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dashadas_perception.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

if(DASHADAS_BUILD_EXAMPLES)
  add_executable(dashadas_sample_client examples/sample_client.cpp)
  target_link_libraries(dashadas_sample_client PRIVATE dashadas_perception)
  target_include_directories(dashadas_sample_client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

  add_executable(dashadas_detect_cli examples/detect_cli.cpp)
  target_link_libraries(dashadas_detect_cli PRIVATE dashadas_perception)
  target_include_directories(dashadas_detect_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(dashadas_detect_cli dashadas_sample_client PROPERTIES
    BUILD_RPATH "$ORIGIN;$ORIGIN/.."
  )

  if(DASHADAS_DOWNLOAD_STB)
    set(STB_IMAGE_H "${CMAKE_CURRENT_BINARY_DIR}/_deps/stb_image.h")
    if(NOT EXISTS "${STB_IMAGE_H}")
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/_deps")
      file(DOWNLOAD
        "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
        "${STB_IMAGE_H}"
        TIMEOUT 30
        STATUS stb_status
      )
      list(GET stb_status 0 stb_code)
      if(NOT stb_code EQUAL 0)
        message(WARNING "stb_image.h download failed; detect_cli will read PPM only")
        file(REMOVE "${STB_IMAGE_H}")
      endif()
    endif()
    if(EXISTS "${STB_IMAGE_H}")
      target_compile_definitions(dashadas_detect_cli PRIVATE DASHADAS_HAS_STB)
      target_include_directories(dashadas_detect_cli PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/_deps)
    endif()
  endif()
endif()
