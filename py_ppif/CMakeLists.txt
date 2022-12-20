cmake_minimum_required(VERSION 3.9)
project(PyPPIF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(pybind11)

pybind11_add_module(pyppif main.cc)
target_link_libraries(pyppif PUBLIC OpenMP::OpenMP_CXX Eigen3::Eigen)
