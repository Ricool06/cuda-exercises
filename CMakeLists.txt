project(CudaExercises LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.10.2)

add_definitions("-std=c++11")

# Initialize Conan
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
CONAN_BASIC_SETUP()

# Create library
add_library(${CMAKE_PROJECT_NAME} vector-addition.cu)
TARGET_LINK_LIBRARIES(${CMAKE_PROJECT_NAME} ${CONAN_LIBS})

# Install library
INSTALL(TARGETS ${CMAKE_PROJECT_NAME}
        DESTINATION lib)