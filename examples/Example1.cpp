/**
 * A Nonlinear Least Squares Solver Example
 *
 * Problem statement: Example 1 contained in this executable is a simple test
 * of the nonlinear least squares solver(s) on a linear objective function.
 */
#include "GaussNewton.h"
#include "GradientDescent.h"
#include "SolverOpts.h"
#include "functors/Example1Functor.h"
#include <iostream>
#include <random>

int main() {
  // Create a functor for the example 1 problem
  std::shared_ptr<ModelFunctor> modelPtr = std::make_shared<Example1Functor>();

  // Define the truth model parameters
  Eigen::VectorXd ATruth(4);
  ATruth << 10.0, -20.0, 30.0, -40.0;

  // Random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dist(0.0, 0.5);

  // Generate simulated data
  double start = 0.0;
  double end = 10.0;
  int numPoints = 100;
  Eigen::VectorXd X(numPoints);
  Eigen::VectorXd Y(numPoints);
  for (int i = 0; i < numPoints; i++) {
    X(i) = start + i * (end - start) / (numPoints - 1);
    Y(i) = (*modelPtr)(ATruth, X(i)) + dist(gen); // TODO: Add noise
  }

  // Initialize model parameter guesses
  Eigen::VectorXd A(4);
  A << 11.8, -13.8, 25.0, -50.0;

  // Set the solver options
  SolverOpts optsGD = {
      .max_iter = 1000, .convergence_criterion = 1.0e-6, .alpha = 0.025};
  SolverOpts optsGN = {
      .max_iter = 1000, .convergence_criterion = 1.0e-6, .alpha = 1.0};

  // Create a Gradient Descent Nonlinear Optimizer
  GradientDescent gd(modelPtr, A, X, Y, optsGD);
  if (gd.isInitialized()) {
    std::cout << "Gradient Descent Model Successfully Initialized!"
              << std::endl;
  }

  // Create a Gauss-Newton Nonlinear Optimizer
  GaussNewton gn(modelPtr, A, X, Y, optsGN);
  if (gn.isInitialized()) {
    std::cout << "Gauss-Newton Model Successfully Initialized!" << std::endl;
  }

  // Run optimizations
  bool successGD = gd.optimize();
  bool successGN = gn.optimize();

  return 0;
}
