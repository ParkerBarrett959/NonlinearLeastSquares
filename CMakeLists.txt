############################################################
#      Cmake Build of Nonlinear Least Squares Solver       #
############################################################

# Project Setup
cmake_minimum_required(VERSION 3.22.0)
project(nonlinear_least_squares)
set(CMAKE_CXX_STANDARD 17)

# Find Eigen
find_package(Eigen3 REQUIRED)

# Add Header List
set(HEADER_LIST ${PROJECT_SOURCE_DIR}/include/NonlinearOptimizer.h
                ${PROJECT_SOURCE_DIR}/include/ModelFunctor.h)

# Add Library
add_library(nonlinear_least_squares INTERFACE)

# Target Include Directories
target_include_directories(nonlinear_least_squares INTERFACE ${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIRS} )

# Example Case Build
add_executable( example1 ${PROJECT_SOURCE_DIR}/examples/Example1.cpp ${HEADER_LIST})
target_link_libraries(
  example1
  nonlinear_least_squares
)

# Target Include Directories
target_include_directories(example1 PUBLIC ${PROJECT_SOURCE_DIR}/include ${EIGEN_INCLUDE_DIRS})
