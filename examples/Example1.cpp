/**
 * A Nonlinear Least Squares Solver Example
 *
 * Problem statement: Example 1 contained in this executable is a simple test
 * of the nonlinear least squares solver(s) on a linear objective function.
 */
#include "GaussNewton.h"
#include "GradientDescent.h"
#include "LevenbergMarquardt.h"
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
  double end = 100.0;
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
  SolverOpts optsGD = {.max_iter = 50000,
                       .convergence_criterion = 1.0e-6,
                       .print_steps = false,
                       .alpha = 1.0e-9,
                       .lambda0 = 1.0,
                       .factor = 2.0};
  SolverOpts optsGN = {.max_iter = 1000,
                       .convergence_criterion = 1.0e-6,
                       .print_steps = false,
                       .alpha = 1.0,
                       .lambda0 = 1.0,
                       .factor = 2.0};
  SolverOpts optsLM = {.max_iter = 1000,
                       .convergence_criterion = 1.0e-6,
                       .print_steps = false,
                       .alpha = 1.0,
                       .lambda0 = 1.0,
                       .factor = 2.0};

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

  // Create a Levenberg-Marquardt Nonlinear Optimizer
  LevenbergMarquardt lm(modelPtr, A, X, Y, optsLM);
  if (!lm.isInitialized()) {
    return 1;
  }

  // Run optimizations
  bool successGD = gd.optimize();
  bool successGN = gn.optimize();
  bool successLM = lm.optimize();

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
  std::cout << "\nLevenberg-Marquardt:" << std::endl;
  if (lm.optimizationConverged()) {
    std::cout << "    Converged: True" << std::endl;
  } else {
    std::cout << "    Converged: False" << std::endl;
  }
  std::cout << "    Number of Steps Run: " << lm.getNumberSteps() << std::endl;
  std::cout << "    Final Parameters = " << lm.getModelParameters().transpose()
            << std::endl;
  std::cout << "    Error = " << (lm.getModelParameters() - ATruth).transpose()
            << std::endl;
  std::cout << "    Error Magnitude = "
            << (lm.getModelParameters() - ATruth).norm() << std::endl;
  return 0;
}
