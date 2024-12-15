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

  // Define the truth model parameters
  Eigen::Vector3d ATruth{3.0, 2.0, 1.0};
  
  // Generate simulated data
  double start = 0.0;
  double end = 10.0;
  int numPoints = 100;
  Eigen::VectorXd X(numPoints);
  Eigen::VectorXd Y(numPoints);
  for (int i = 0; i < numPoints; i++) {
    Eigen::VectorXd xCurr(1);
    xCurr(0) = start + i * (end - start) / (numPoints - 1); 
    X(i) = xCurr(0);
    Y(i) = (*modelPtr)(ATruth, xCurr) + 0.0; // TODO: Add noise
  }

  // Initialize model parameter guesses
  Eigen::Vector3d A = Eigen::Vector3d::Zero();

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