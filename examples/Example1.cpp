/**
 * A Nonlinear Least Squares Solver Example
 *
 * Problem statement: Example 1 contained in this executable is a simple test
 * of the nonlinear least squares solver(s) on a linear objective function.
 */
#include "GradientDescent.h"
#include "SolverOpts.h"
#include "functors/Example1Functor.h"
#include <iostream>

int main() {
  // Create a functor for the example 1 problem
  std::shared_ptr<ModelFunctor> modelPtr = std::make_shared<Example1Functor>();

  // Define the truth model parameters
  Eigen::Vector3d ATruth{3.0, 2.0, 1.0};

  // Generate simulated data
  double start = 0.0;
  double end = 10.0;
  int numPoints = 100;
  Eigen::VectorXd X(numPoints);
  Eigen::VectorXd Y(numPoints);
  for (int i = 0; i < numPoints; i++) {
    X(i) = start + i * (end - start) / (numPoints - 1);
    Y(i) = (*modelPtr)(ATruth, X(i)) + 0.0; // TODO: Add noise
  }

  // Initialize model parameter guesses
  Eigen::Vector3d A = Eigen::Vector3d::Zero();

  // Use the default solver options
  SolverOpts opts;

  // Create a Gradient Descent Nonlinear Optimizer
  GradientDescent gd(modelPtr, A, X, Y, opts);
  if (gd.isInitialized()) {
    std::cout << "Gradient Descent Model Successfully Initialized!"
              << std::endl;
  }

  // Run optimization
  bool success = gd.optimize();

  return 0;
}
