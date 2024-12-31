/**
 * A Nonlinear Least Squares Solver Example
 *
 * Problem statement: Example 2 contained in this executable is a test of the
 * nonlinear least squares solver(s) on a nonlinear objective function.
 */
#include "GaussNewton.h"
#include "GradientDescent.h"
#include "SolverOpts.h"
#include "functors/Example2Functor.h"
#include <iostream>
#include <random>

int main() {
  // Create a functor for the example 2 problem
  std::shared_ptr<ModelFunctor> modelPtr = std::make_shared<Example2Functor>();

  // Define the truth model parameters
  Eigen::VectorXd ATruth(4);
  ATruth << 20.0, 10.0, 1.0, 50.0;

  // Random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dist(0.0, 0.5);

  // Generate simulated data
  double start = 0.0;
  double end = 100.0;
  int numPoints = 1000;
  Eigen::VectorXd X(numPoints);
  Eigen::VectorXd Y(numPoints);
  for (int i = 0; i < numPoints; i++) {
    X(i) = start + i * (end - start) / (numPoints - 1);
    Y(i) = (*modelPtr)(ATruth, X(i)) + dist(gen);
  }

  // Initialize model parameter guesses
  Eigen::VectorXd A(4);
  A << 9.1, 11.8, 8.7, 98.6;

  // Set the solver options
  SolverOpts optsGD = {.max_iter = 25000,
                       .convergence_criterion = 1.0e-6,
                       .alpha = 3.2e-6,
                       .print_steps = false};
  SolverOpts optsGN = {.max_iter = 1000,
                       .convergence_criterion = 1.0e-6,
                       .alpha = 1.0,
                       .print_steps = false};

  // Create a Gradient Descent Nonlinear Optimizer
  GradientDescent gd(modelPtr, A, X, Y, optsGD);
  if (!gd.isInitialized()) {
    return 1;
  }

  // Create a Gauss-Newton Nonlinear Optimizer
  GaussNewton gn(modelPtr, A, X, Y, optsGN);
  if (!gn.isInitialized()) {
    return 1;
  }

  // Run optimizations
  bool successGD = gd.optimize();
  bool successGN = gn.optimize();

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
  std::cout << "\nGauss Newton:" << std::endl;
  if (gn.optimizationConverged()) {
    std::cout << "    Converged: True" << std::endl;
  } else {
    std::cout << "    Converged: False" << std::endl;
  }
  std::cout << "    Number of Steps Run: " << gn.getNumberSteps() << std::endl;
  std::cout << "    Final Parameters = " << gn.getModelParameters().transpose()
            << std::endl;
  std::cout << "    Error = " << (gn.getModelParameters() - ATruth).transpose()
            << std::endl;
  std::cout << "    Error Magnitude = "
            << (gn.getModelParameters() - ATruth).norm() << std::endl;
  return 0;
}
