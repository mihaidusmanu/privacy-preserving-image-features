cmake_minimum_required(VERSION 3.9)
project(PyPPIF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
find_package(Eigen3 REQUIRED)

add_subdirectory(pybind11)

pybind11_add_module(pyppif main.cc)
target_link_libraries(pyppif PUBLIC Eigen3::Eigen)

# OpenMP only parallelises the exhaustive matchers. Apple Clang ships without
# it, so build single-threaded rather than failing when it is not found.
if(OpenMP_CXX_FOUND)
  target_link_libraries(pyppif PUBLIC OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found: the pyppif matchers will run single-threaded.")
endif()
