cmake_minimum_required(VERSION 3.10)
project(inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(cppflow REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

add_executable(inference
    main.cpp
    src/inferencer.cpp
)

target_include_directories(inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(inference PRIVATE
    cppflow::cppflow
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(inference PRIVATE -O3 -march=native)
endif()
