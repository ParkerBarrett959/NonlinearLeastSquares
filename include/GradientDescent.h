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
   * @param A An n dimensional vector of parameters to be estimated in the least squares
   * problem
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
    std::cout << "Gradient Descent optimization" << std::endl;
    std::cout << "------------------------------------------------"
              << std::endl;
    // Loop until convergence
    bool converged = false;
    int iter = 0;
    while (!converged && iter < opts_.max_iter) {
      // Increment the iteration count
      iter += 1;

      // Compute the Jacobian of the Cost Function
      Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, A_.size());
      for (int i = 0; i < Y_.size(); i++) {
        // Compute Model Prediction Error
        double e = (*mModelFunctor)(A_, X_.row(i)) - Y_(i);

        // Compute Jacobians
        J = J + (*mJacobianFunctor)(A_, X_.row(i), e);
      }
      J = J / Y_.size();
      
      // Compute the gradient descent step
      Eigen::MatrixXd hgd = opts_.alpha * J;

      // Update the model parameters
      A_ = A_ - hgd.transpose();

      // Print Current Iteration
      std::cout << "Iter = " << iter << std::endl;
      std::cout << "    Parameters = " << A_.transpose() << std::endl;
      std::cout << "    Step Size = " << hgd.norm() << std::endl;

      // Check for convergence
      if (hgd.norm() < opts_.convergence_criterion) {
        std::cout << "Gradient descent converged!" << std::endl;
        converged = true;
      }
    }
    return true;
  }
};

#endif // GRADIENT_DESCENT_H