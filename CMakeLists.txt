cmake_minimum_required(VERSION 3.10)
project(ObjectDetectionApp)

# Specify the build type
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++17 -O3 -pthread -g")

# Set the policy for OpenGL library preference
if(POLICY CMP0072)
  cmake_policy(SET CMP0072 NEW)
endif()

# Find OpenCV package
find_package(OpenCV 4.1 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(InferenceEngine REQUIRED)

# set imgui directory
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)
# set glad directory
set(GLAD_DIR ${CMAKE_SOURCE_DIR}/third_party/glad)
# set yaml-cpp directory
set(YAML_CPP_DIR ${CMAKE_SOURCE_DIR}/third_party/yaml-cpp)

# Add GLAD as a library
add_library(glad ${GLAD_DIR}/src/gl.c)

# include directories
include_directories(include)
include_directories(include/interfaces)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${YAML_CPP_DIR}/include)
# Add the OpenVINO include directories using find_package
if(OpenVINO_FOUND)
    include_directories(${InferenceEngine_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenVINO package not found.")
endif()

# Add the path to imgui
include_directories(${IMGUI_DIR})
include_directories(${GLAD_DIR}/include)

# link directories
link_directories(${OpenCV_LIBRARY_DIRS})

add_definitions(${OpenCV_DEFINITIONS})

# Add the path to ImGui source files
file(GLOB IMGUI_SOURCES "${IMGUI_DIR}/*.cpp")
# Add ImGui backend source files
set(IMGUI_IMPL_SOURCES
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
# Specify YAML-CPP source files
file(GLOB YAML_CPP_SOURCES ${YAML_CPP_DIR}/src/*.cpp)

add_executable(object_detector src/main.cpp
                               src/OpenCVFrameCollector.cpp
                               src/OpenCVFramePreprocessor.cpp
                               src/OpenCVDnnInferenceEngine.cpp
                               src/OpenCVFramePostprocessor.cpp
                               src/OpenGLBackend.cpp
                               src/Drawer.cpp
                               src/OpenVINOFramePreprocessor.cpp
                               src/OpenVINOInferenceEngine.cpp
                               src/OpenVINOFramePostprocessor.cpp
                               ${IMGUI_SOURCES}
                               ${IMGUI_IMPL_SOURCES}
                               ${YAML_CPP_SOURCES})

target_link_libraries(object_detector
    ${OpenCV_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${InferenceEngine_LIBRARIES}
    glfw
    glad
    dl)
