project(cuda-exercises-tests LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.10.2)

add_definitions("-std=c++11")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(${PROJECT_NAME} vector-addition.cu)
target_link_libraries(${PROJECT_NAME} ${CONAN_LIBS})

enable_testing()
add_test(NAME some-test
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
         COMMAND ${PROJECT_NAME})