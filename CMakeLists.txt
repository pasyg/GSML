cmake_minimum_required(VERSION 3.30)
project(gsml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Boost REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(gsml_nn)
target_sources(gsml_nn
    PUBLIC 
    FILE_SET CXX_MODULES
    BASE_DIRS
    FILES
        src/nn/layer.cpp
        src/nn/conv.cpp
        src/nn/dense.cpp
        src/nn/nn.cpp
        src/nn/pooling.cpp

        src/nn/activation/relu.cpp
        src/nn/activation/sigmoid.cpp
        src/nn/activation/tanh.cpp

        src/nn/loss/mse.cpp
)

add_library(gsml_nn_optimizer)
target_sources(gsml_nn_optimizer
    PUBLIC
    FILE_SET CXX_MODULES
    BASE_DIRS
    FILES
        src/nn/optimizer/adam.cpp
        src/nn/optimizer/optimizer.cpp
)


add_executable(gsml main.cpp)
target_link_libraries (gsml Eigen3::Eigen 
                            gsml_nn
                            gsml_nn_optimizer)