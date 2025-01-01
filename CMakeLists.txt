############################################################
#      Cmake Build of Nonlinear Least Squares Solver       #
############################################################

# Project Setup
cmake_minimum_required(VERSION 3.22.0)
project(nonlinear_least_squares)
set(CMAKE_CXX_STANDARD 17)

# Find Eigen
find_package(Eigen3 REQUIRED)

# Add Library Header List
set(HEADER_LIST ${PROJECT_SOURCE_DIR}/include/NonlinearOptimizer.h
                ${PROJECT_SOURCE_DIR}/include/ModelFunctor.h
                ${PROJECT_SOURCE_DIR}/include/GradientDescent.h
                ${PROJECT_SOURCE_DIR}/include/GaussNewton.h
                ${PROJECT_SOURCE_DIR}/include/LevenbergMarquardt.h
                ${PROJECT_SOURCE_DIR}/include/SolverOpts.h)

# Add Library
add_library(nonlinear_least_squares INTERFACE)

# Target Include Directories
target_include_directories(nonlinear_least_squares INTERFACE ${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIRS} )

# Add Example Header List
set(EXAMPLE_HEADER_LIST ${PROJECT_SOURCE_DIR}/examples/functors/Example1Functor.h
                        ${PROJECT_SOURCE_DIR}/examples/functors/Example2Functor.h
                        ${PROJECT_SOURCE_DIR}/examples/functors/Example3Functor.h)

# Example 1 Case Build
add_executable( example1 ${PROJECT_SOURCE_DIR}/examples/Example1.cpp ${HEADER_LIST} ${EXAMPLE_HEADER_LIST})
target_link_libraries(
  example1
  nonlinear_least_squares
)

# Example 2 Case Build
add_executable( example2 ${PROJECT_SOURCE_DIR}/examples/Example2.cpp ${HEADER_LIST} ${EXAMPLE_HEADER_LIST})
target_link_libraries(
  example2
  nonlinear_least_squares
)

# Example 3 Case Build
add_executable( example3 ${PROJECT_SOURCE_DIR}/examples/Example3.cpp ${HEADER_LIST} ${EXAMPLE_HEADER_LIST})
target_link_libraries(
  example3
  nonlinear_least_squares
)

# Target Include Directories
target_include_directories(example1 PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples ${EIGEN_INCLUDE_DIRS})
target_include_directories(example2 PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples ${EIGEN_INCLUDE_DIRS})
target_include_directories(example3 PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples ${EIGEN_INCLUDE_DIRS})