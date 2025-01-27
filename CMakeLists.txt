cmake_minimum_required(VERSION 3.30)
project(MNIST_DNN_Part2)

set(CMAKE_CXX_STANDARD 26)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

add_executable(MNIST_DNN_Part2 mnist_dnn_optimized.cpp)

