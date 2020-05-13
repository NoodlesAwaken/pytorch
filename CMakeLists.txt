cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(pytorch pytorch.cpp)
target_link_libraries(pytorch "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET pytorch PROPERTY CXX_STANDARD 14)
