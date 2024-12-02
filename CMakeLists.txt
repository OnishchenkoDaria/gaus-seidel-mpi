cmake_minimum_required(VERSION 3.31)
project(dc_lab6)

set(CMAKE_CXX_STANDARD 20)

# Locate MPI libraries and headers
find_package(MPI REQUIRED)

# Add executable and link MPI libraries
add_executable(ParallelGS ParallelGS.cpp)
add_executable(SerialGS SerialGS.cpp)
target_link_libraries(ParallelGS PRIVATE MPI::MPI_CXX)