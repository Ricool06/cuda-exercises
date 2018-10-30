project(cuda-exercises LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.10.2)

add_definitions("-std=c++11")

# Initialize Conan
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
CONAN_BASIC_SETUP()
enable_testing()

## LIBRARY ##
include(GoogleTest)
include_directories(lib)
include_directories(test)
link_directories(lib)
link_directories(test)

# Instead of having to add source filenames to this
# file every time we create a new source, just glob it.
# This comes with the caveat that you'll have to rerun
# the build.
file(GLOB LIB_SOURCES lib/*.cu lib/*.cuh)
add_library(${CMAKE_PROJECT_NAME} ${LIB_SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${CONAN_LIBS})

## END LIBRARY ##

## TESTS ##
# Create test executable
set(TEST_EXECUTABLE_NAME ${PROJECT_NAME}-tests)
file(GLOB TEST_SOURCES test/*.cu)
add_executable(${TEST_EXECUTABLE_NAME} test/vector-addition.cu)
# Link necessary directories
target_link_libraries(${TEST_EXECUTABLE_NAME} ${CONAN_LIBS} ${CMAKE_PROJECT_NAME})

# Add test executable to runnable tests
gtest_discover_tests(${TEST_EXECUTABLE_NAME}
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
## END TESTS ##
