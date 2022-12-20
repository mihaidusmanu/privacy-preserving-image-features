cmake_minimum_required(VERSION 3.9)
project(PyPPIFCUDA LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

add_subdirectory(pybind11)

add_library(cuda_matching STATIC lib/matching.cu)
set_target_properties(cuda_matching PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(pyppifcuda main.cc)
target_link_libraries(pyppifcuda PUBLIC Eigen3::Eigen cuda_matching)
