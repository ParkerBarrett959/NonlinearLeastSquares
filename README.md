# Nonlinear Least Squares

A C++ header-only library and example executables for solving single-input single-output nonlinear least squares problems. The library includes support for Gradient Descent, Gauss-Newton and Levenberg-Marquardt solvers and a simple interface for defining and solving custom nonlinear functions.

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
After building, run the examples with
```
# Run example 1
./example1

# Run Example 2
./example2
```
# Nonlinear Least Squares Problem Formulation

Consider a single-input, single-output system of the form $\hat{y}(x,\mathbf{a})$, where x is a scalar independent variable, $\mathbf{a}$ is a vector of n unknown model parameters, $\mathbf{a} = [a_1, a_2, ..., a_n]$, and y is the scalar, dependent variable output. Given a set of m data points, $\mathbf{y} = [y_1, y_2, ..., y_m]$, the nonlinear least squares problem is to find the optimal set of parameters, $\mathbf{a}$, which minimize the sum of the squares of the residuals between the data points and the model. The cost function is given by

$$
J = \frac{1}{2} \sum_{i=1}^m [y_i - \hat{y}(x_i, \mathbf{a})]^2
$$

If the function, $\hat{y}(x, \mathbf{a})$ is linear with respect to the model coefficients $\mathbf{a}$, the cost function can be minimized in a single step using a linear least squares technique. This codebase focuses on solving systems which are nonlinear in the model parameters, therefore iterative methods are required. In general, iterative methods involve finding perturbations, $\Delta{\mathbf{a}} = \mathbf{h}$, to the model parameters, which take the parameters closer and closer to the optimal values. A standard step in the algorithm takes the form

$$
a_{i+1} = a_{i} - \Delta{\mathbf{a}} = a_{i} - \mathbf{h}
$$

The three solvers implemented in this codebase, each of which are used to determine the model parameter perturbations, are Gradient Descent, the Gauss-Newton Method and the Levenberg-Marquardt Method. Details of each of these algorithms can be found in the sections to follow.

One last note of importance relates to the gradient of the cost function, a term used frequently by each algorithm. The gradient is a multivariate derivative which represents the direction and magnitude of the rate of change of cost function at a given point. Using the chain rule, the gradient of the cost function can be found by taking the gradient of $J$ with respect to the model parameters, $\mathbf{a}$.

$$
\frac{\partial{J}}{\partial{\mathbf{a}}} = - \sum_{i=1}^m [y_i - \hat{y}(x_i, \mathbf{a})](\frac{\partial{y(x_i, \mathbf{a})}}{\partial{\mathbf{a}}})
$$

This expression is relevant for all problems which have the least squares cost function described above. Notice how the final term involves taking the partial derivative of the nonlinear model with respect to the parameters. This will be different for each unique model and must be computed, theoretically or numerically. In this codebase, the model functors expect the gradients to be provided along with the base model.

# Gradient Descent Algorithm

The Gradient Descent algorithm is conceptually straightforward. If the gradient of the cost function with respect to the model parameters, $\frac{\partial{J}}{\partial{\mathbf{a}}}$, represents the sensitivity of the cost to changes in the parameters, then we can compute the gradient at each step, find the direction of steepest descent (minimizing the cost function), and continually step in that direction until the cost can not shrink any further.

In particular, the Gradient Descent perturbation is given by

$$
h_{GD} = \alpha \frac{\partial{J}}{\partial{\mathbf{a}}}\Bigr|_{\mathbf{a}}
$$

In the expression above, $\alpha$ represents the step size, commonly referred to as the learning rate, and the gradient expression is evaluated at the current model parameter values. The learning rate is typically set to some value less than 1 (0.1 for example), which allows the algorithm to converge.

While Gradient Descent is extremely powerful, it is not always guaranteed to converge to the optimal (or any) solution. If the cost function is non-convex for example, Gradient Descent may converge to a local optimum, but miss a global optimum, or fail to converge entirely. The algorithm tends to perform best further away from the optimal solution with relatively steep gradients, and worst in shallower section of the cost function. This leads to relatively quick convergence when you are farther away from the solution, but the slower convergence near the optimal solution.

# Gauss-Newton Algorithm

The Gauss-Newton is a modification of the widely used Newton-Raphson root-finding technique, in which an analytical expression for the Hessian is required. The Newton-Raphson method is given by

$$
a_{i+1} = a_{i} - \Delta{\mathbf{a}} = a_{i} - (\mathbf{H})^{-1} \frac{\partial{J}}{\partial{\mathbf{a}}}\Bigr|_{\mathbf{a}}
$$

In many nonlinear least squares problems, an exact expression for the Hessian can not be provided, or is too numerically expensive to compute. In the Gauss-Newton method, a first-order approximation of the Hessian is made as

$$
\mathbf{H_{\textrm{approx}}} \approx \sum_{i=1}^m Y^{T}Y
$$

where

$$
Y = \frac{\partial{y(x_i, \mathbf{a})}}{\partial{\mathbf{a}}}\Bigr|_{\mathbf{a}}
$$

The final Gauss-Newton perturbation then becomes

$$
h_{GN} = \mathbf{H_{\textrm{approx}}^{-1}} \frac{\partial{J}}{\partial{\mathbf{a}}}\Bigr|_{\mathbf{a}}
$$

Notice the similarities to the perturbation expression from the Gradient Descent solver. The inverse of the Hessian can be thought of as a better estimate of the learning rate, $\alpha$. The Gauss-Newton method typically performs better than Gradient Descent, without the need for parameter tuning, in particular in "shallow" sections of the optimization problem near the solution.

# Levenberg-Marquardt Algorithm

The Levenberg-Marquardt algorithm can be thought of as a combination of Gradient Descent and Gauss-Newton, taking the strengths of each. While many formulations of Levenberg-Marquardt exist, the version used in this codebase uses the following perturbation:

$$
h_{LM} = \mathbf{(H_{\textrm{approx}}+ \lambda \mathbb{I})^{-1}} \frac{\partial{J}}{\partial{\mathbf{a}}}\Bigr|_{\mathbf{a}}
$$

Notice the similarity to the Gauss-Newton perturbation. In particular, as $\lambda$ trends towards zero, $h_{LM} = h_{GN}$. Conversely, as $\lambda$ increases, the magnitude of the resulting matrix inversion shrinks, which has the effect of shortening the step size in the same manner as the learning rate, $\alpha$, in the Gradient Descent algorithm. Since changing the value of $\lambda$ can shift our algorithm to behave like Gauss-Newton or Gradient Descent, it seems reasonable that we could set the parameter in such a way to maximize the strengths of each.

In practice, $\lambda$, is initialized to some value, $\lambda_{0}$. At each iteration, we then compute the Levenberg-Marquardt step using the expression above with $\lambda = \lambda$ and $\lambda = \frac{\lambda}{\nu}$. $\nu$ here is scaling parameter, set to some value $\nu > 1$.

Next, we evaluate the cost function at each of these two possible perturbations. If neither perturbation shows improvement over the current model parameters, we simply set $\lambda = \lambda \nu$ and repeat the procedure. By increasing $\lambda$, this shifts the perturbation closer to a Gradient Descent step with a smaller learning rate.

If either of the two perturbations shows improvement over the existing score, we then make a decision about which perturbation to use. If the smaller value, $\lambda = \frac{\lambda}{\nu}$ showed an improvement, we use this one and set $\lambda = \frac{\lambda}{\nu}$. By shrinking $\lambda$, we shift the problem closer towards Gauss-Newton, which typically converges faster closer to the solution.

If only the larger value, $\lambda = \lambda$ demonstrated an improvement, we simply use this perturbation and continue to the next iteration without modifying $\lambda$.

While Levenberg-Marquardt will not always outperform Gradient-Descent and Gauss-Newton, it does have several advantages, most notably its robustness, If the Hessian matrix is poorly conditioned, Gauss-Newton can fail or converge slowly, which the damping coefficient in Levenberg-Marquardt helps prevent. Additionally, if the model parameters begin far from the optimal solution, Levenberg-Marquardt will shift closer to Gradient Descent, which is less aggressive and helps pull the model parameters closer to the optimum slowly, before shifting towards the more aggressive, but faster convergence of Gauss-Newton. 

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

  // Random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dist(0.0, 0.5);

  // Generate simulated, noisy data - see below as an example
  double start = 0.0;
  double end = 100.0;
  int numPoints = 100;
  Eigen::VectorXd X(numPoints);
  Eigen::VectorXd Y(numPoints);
  for (int i = 0; i < numPoints; i++) {
    X(i) = start + i * (end - start) / (numPoints - 1);
    Y(i) = (*modelPtr)(ATruth, X(i)) + dist(gen);
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

  // Get model parameters and print results
  std::cout << "\n--------------------------------------------\n" << std::endl;
  std::cout << "Truth Parameters: " << ATruth.transpose() << std::endl;
  std::cout << "Gradient Descent:" << std::endl;
  if (gd.optimizationConverged()) {
    std::cout << "    Converged: True" << std::endl;
  } else {
    std::cout << "    Converged: False" << std::endl;
  }
  std::cout << "    Number of Steps Run: " << gd.getNumberSteps() << std::endl;
  std::cout << "    Final Parameters = " << gd.getModelParameters().transpose()
            << std::endl;
  std::cout << "    Error = " << (gd.getModelParameters() - ATruth).transpose()
            << std::endl;
  std::cout << "    Error Magnitude = "
            << (gd.getModelParameters() - ATruth).norm() << std::endl;
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
- Gavin, H. P. (2024, May 5). The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.
- https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
