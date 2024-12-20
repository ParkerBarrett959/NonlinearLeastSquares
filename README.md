# Nonlinear Least Squares Solver
A small C++ header-only library and example executables for solving single-input single-output nonlinear least squares problems. The library includes support for Gradient Descent, Gauss-Newton and Levenberg-Marquardt solvers and a simple interface for defining and solving your own nonlinear functions.

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
# Nonlinear Least Squares Problem Formulation

Consider a single-input, single-output system of the form $\hat{y}(x,\mathbf{a})$, where x is a scalar independent variable, $\mathbf{a}$ is a vector of n unknown model parameters, $\mathbf{a} = [a_1, a_2, ..., a_n]$, and y is the scalar, dependent variable output. Given a set of m data points, $\mathbf{y} = [y_1, y_2, ..., y_m]$, the nonlinear least squares problem is to find the optimal set of parameters, $\mathbf{a}$, which minimize the sum of the squares of the residuals between the the data points and the model. The cost function is given by

$$
J = \sum_{i=1}^m [y_i - \hat{y}(x_i, \mathbf{a})]^2
$$

If the function, $\hat{y}(x, \mathbf{a})$ is linear with respect to the model coefficients $\mathbf{a}$, the cost function can be minimized in a single step using a linear least squares technique. Thise codebase focuses on solving systems which are nonlinear in the model parameters, therefore iterative methods are needed. In general, iterative methods involve finding perturbations, $\Delta{\mathbf{a}} = \mathbf{h}$, to the model parameters, which take the parameters closer and closer to the optimal values. A standard step in the algorithm takes the form

$$
a_{i+1} = a_{i} + \Delta{\mathbf{a}} = a_{i} + \mathbf{h}
$$

The three solvers implemented in this codebase, each of which are used to determine the model parameter perturbations, are Gradient Descent, the Gauss-Newton Method and the Levenberg-Marquardt Method. Details of each of these algorithms can be found in the sections to follow.

One last note of importance relates to the gradient of the cost function, a term used frequently by each algorithm. The gradient is a multivariate derivative which represents the direction and magnitude of the rate of change of cost function at a given point. Using the chain rule, the gradient of the cost function can be found by taking the gradient of $J$ with respect to the model parameters, $\mathbf{a}$.

$$
\frac{\partial{J}}{\partial{\mathbf{a}}} = -2 \sum_{i=1}^m [y_i - \hat{y}(x_i, \mathbf{a})](\frac{\partial{y(x_i, \mathbf{a})}}{\partial{\mathbf{a}}})
$$

This expression holds for all problems with of the form and with the cost functions described above. Notice how the final term involves taking the partial derivative of the model with respect to the parameters. This will change for each different nonlinear model and must be computed, theoretically or numerically.

# Gradient Descent Algorithm
Insert
# Gauss-Newton Algorithm
TODO
# Levenberg-Marquardt Algorithm
TODO
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
# Sources
Gavin, H. P. (2024, May 5). The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.
