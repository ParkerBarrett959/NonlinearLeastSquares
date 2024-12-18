# Nonlinear Least Squares Solver
A small C++ header-only library and example executables for solving single-input single-output nonlinear least squares problems. The library includes support for Gradient Descent, Gauss-Newton and Levenberg-Marquardt solver and a simple interface for defining and solving your own nonlinear functions.

# Dependencies
* C++ 11 (or greater) <br />
* CMake (3.22.0 or greater) <br />
* Eigen (3.3 or greater) <br />
# Build
```
mkdir build
cd build
cmake ..
make
```
# Run Examples
To modify the solver options and parameters, modify ```include/SolverOpts.h```. After building, run the examples with
```
# Run example 1
./example1
```
# Solving Custom Nonlinear Least Squares Problems

This library provides a convenient mechanism for defining and solving your own least squares problems. You will need to create your own nonlinear function and setup an executable to run the solver. To create the nonlinear function, create a new file in the ```examples/functors/``` directory. This functor inherits from the base model functor class and must implement the () operator and gradient function. Use the following code as a template:
```
#ifndef CUSTOM_FUNCTOR_H
#define CUSTOM_FUNCTOR_H

#include "ModelFunctor.h"
#include <Eigen/Dense>

/**
 * Custom functor
 *
 * @brief: This functor implements some custom nonlinear function.
 */
class CustomFunctor : public ModelFunctor {
public:
  /**
   * Functor operator
   *
   * @param a A vector containing the model parameters that are being solved for
   * @param x A double representing the current independent variables to
   * evaluate the model at
   * @return A double representing the model output value
   */
  double operator()(const Eigen::VectorXd &a, const double x) override {
    double y; // define y here as a function of your parameters [a1, a2, ..., an] and the independent variable, x
    return y;
  }

  /**
   * Gradient of the functor
   *
   * @param a A vector containing the model parameters that are being solved for
   * @param x A double representing the current independent variables to
   * evaluate the model at
   * @return A vector representing the gradient of the model wrt the parameters
   */
  Eigen::VectorXd gradient(const Eigen::VectorXd &a, const double x) override {
    Eigen::VectorXd dyda(a.size());
    // Set the elements of dyda to the gradient of the function y wrt the parameters a
    return dyda;
  }
};
#endif // CUSTOM_FUNCTOR_H
```
After setting your functor, create an executable in the ```examples/``` directory. This executable will be the main entry point for your code and is used to configure the system. The following code can be used as a template.
```
/**
 * CustomProblem.cpp
 */
#include "GradientDescent.h" // swap this out for another solver if desired
#include "SolverOpts.h"
#include "functors/CustomFunctor.h"
#include <iostream>

int main() {
  // Create a functor for the custom problem
  std::shared_ptr<ModelFunctor> modelPtr = std::make_shared<CustomFunctor>();

  // Define the truth model parameters
  Eigen::VectorXd ATruth; // Set ATruth values here

  // Generate simulated, noisy data - see below as an example
  double start = 0.0;
  double end = 10.0;
  int numPoints = 100;
  Eigen::VectorXd X(numPoints);
  Eigen::VectorXd Y(numPoints);
  for (int i = 0; i < numPoints; i++) {
    X(i) = start + i * (end - start) / (numPoints - 1);
    Y(i) = (*modelPtr)(ATruth, X(i)) + 0.0; // Add noise here if desired
  }

  // Initialize model parameter guesses
  Eigen::VectorXd A; // Make sure the model parameters initial guess matches the size of the truth parameters

  // Set solver options
  SolverOpts opts; // This uses default - modify if desired

  // Create an optimizer - use a different solver if desired
  GradientDescent gd(modelPtr, A, X, Y, opts);
  if (gd.isInitialized()) {
    std::cout << "Gradient Descent Model Successfully Initialized!"
              << std::endl;
  }

  // Run optimization
  bool success = gd.optimize();

  return 0;
}
```
Next, to build your example, modify the ```CMakeLists.txt``` file in the root directory of the repository. You can add the following block to the end of the file.

```
# Add Example Header List
set(CUSTOM_HEADER_LIST ${PROJECT_SOURCE_DIR}/examples/functors/CustomFunctor.h)

# Example Case Build
add_executable( CustomProblem ${PROJECT_SOURCE_DIR}/examples/CustomProblem.cpp ${HEADER_LIST} ${CUSTOM_HEADER_LIST})
target_link_libraries(
  Custom
  nonlinear_least_squares
)

# Target Include Directories
target_include_directories(CustomProblem PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples ${EIGEN_INCLUDE_DIRS})
```
Finally, rebuild and the new executable, ```CustomProblem``` should be generated. Run your solution with:

```
./CustomProblem
```

# Nonlinear Least Squares Problem Formulation
Insert
# Gradient Descent Algorithm
Insert
# Gauss-Newton Algorithm
TODO
# Levenberg-Marquardt Algorithm
TODO
# Sources
Insert
