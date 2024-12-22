#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include "NonlinearOptimizer.h"

/**
 * Gradient Descent Nonlinear Optimizer
 *
 * @brief: This class inherits from the NonlinearOptimizer interface and
 * implements a gradient descent approach to solving the least squares problem.
 */
class GradientDescent : public NonlinearOptimizer {
public:
  /**
   * c'tor
   *
   * @param model A shared pointer to a functor representing the nonlinear model
   * to be solved.
   * @param A An n dimensional vector of parameters to be estimated in the least
   * squares problem
   * @param X An m dimensional vector of independent variables
   * @param Y An n dimensional vector of dependent variables
   * @param opts The solver options to use
   */
  GradientDescent(std::shared_ptr<ModelFunctor> &model,
                  const Eigen::VectorXd &A, const Eigen::VectorXd &X,
                  const Eigen::VectorXd &Y, const SolverOpts &opts)
      : NonlinearOptimizer(model, A, X, Y, opts) {}

  /**
   * Function to run the gradient descent optimization
   *
   * @return True if the optimization succeeded, false otherwise
   */
  bool optimize() {
    // Initial print statement
    std::cout << "Running Gradient Descent optimization..." << std::endl;

    // Loop until convergence
    while (!optimizerConverged_ && numberSteps_ < opts_.max_iter) {
      // Increment the iteration count
      numberSteps_ += 1;

      // Compute the current cost
      double J = computeJ();

      // Compute the Jacobian of the Cost Function
      Eigen::VectorXd dJdA = computeGradientJ();

      // Compute the gradient descent step
      Eigen::VectorXd hgd = opts_.alpha * dJdA;

      // Update the model parameters
      A_ = A_ - hgd;

      // Check for convergence and print step
      if (hgd.norm() < opts_.convergence_criterion) {
        if (opts_.print_steps) {
          std::cout << "Gradient Descent converged!" << std::endl;
          std::cout << "Number of Iterations: " << numberSteps_ << std::endl;
          std::cout << "Cost: " << J << std::endl;
          std::cout << "Final Step Size: " << hgd.norm() << std::endl;
          std::cout << "Final Model Parameters: " << A_.transpose()
                    << std::endl;
        }
        optimizerConverged_ = true;
      } else {
        if (opts_.print_steps) {
          std::cout << "    i = " << numberSteps_ << ", J = " << J
                    << ", step = " << hgd.norm() << std::endl;
        }
      }
    }
    std::cout << "Gradient Descent Complete!" << std::endl;
    return true;
  }
};

#endif // GRADIENT_DESCENT_H