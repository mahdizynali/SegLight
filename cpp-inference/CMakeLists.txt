cmake_minimum_required(VERSION 3.10)

project(inference)

add_executable(test main.cpp inferencer.hpp)

find_package(cppflow REQUIRED)

find_package(OpenCV REQUIRED)

target_include_directories(
    test PUBLIC
    cppflow::cppflow
)

target_link_libraries(
    test PUBLIC
    cppflow::cppflow
    ${OpenCV_LIBRARIES}
)

target_include_directories(
    test PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
