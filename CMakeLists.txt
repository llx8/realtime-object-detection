cmake_minimum_required(VERSION 3.10)
project(EdgeVisionGateway)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

include_directories(${CMAKE_SOURCE_DIR})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_library(NCNN_LIB ncnn)
if(NOT NCNN_LIB)
    message(FATAL_ERROR "ncnn not found. Install it: https://github.com/Tencent/ncnn")
endif()

set(SRC_FILES
    main.cpp
    pipeline/Pipeline.cpp
    pipeline/stages/CaptureStage.cpp
    pipeline/stages/PreprocessStage.cpp
    pipeline/stages/InferenceStage.cpp
    pipeline/stages/PostprocessStage.cpp
    pipeline/hardware/LinuxV4L2Source.cpp
    pipeline/hardware/v4l2_camera.cpp
    pipeline/utils/FramePool.cpp
)

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} ${NCNN_LIB} pthread)
