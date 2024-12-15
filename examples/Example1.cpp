/**
 * A Nonlinear Least Squares Solver Example
 *
 * Problem statement: Example 1 contained in this executable is a simple test
 * of the nonlinear least squares solver(s) on a linear objective function.
 */
#include "GradientDescent.h"
#include "SolverOpts.h"
#include "functors/Example1Functor.h"
#include "functors/Example1Jacobian.h"
#include <iostream>

int main() {
  // Create a functor for the example 1 problem and the Jacobian
  std::shared_ptr<ModelFunctor> modelPtr = std::make_shared<Example1Functor>();
  std::shared_ptr<ModelJacobian> jacobianPtr =
      std::make_shared<Example1Jacobian>();
  std::make_shared<Example1Jacobian>();

  // Define the model parameters
  Eigen::Vector3d A{0.0, 0.0, 0.0};
  Eigen::Matrix3d X;
  X << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
  Eigen::Vector3d Y{1.0, 2.0, 3.0};

  // Use the default solver options
  SolverOpts opts;

  // Create a Gradient Descent Nonlinear Optimizer
  GradientDescent gd(modelPtr, jacobianPtr, A, X, Y, opts);
  if (gd.isInitialized()) {
    std::cout << "Gradient Descent Model Successfully Initialized!"
              << std::endl;
  }

  // Run optimization
  bool success = gd.optimize();

  return 0;
}