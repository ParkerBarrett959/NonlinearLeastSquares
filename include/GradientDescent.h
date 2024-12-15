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
   * @param jacobian A shared pointer to a functor representing the nonlinear
   * Jacobian of the model
   * @param A A vector of parameters to be estimated in the least squares
   * problem
   * @param X An nxm matrix of model independent variables. Each row corresponds
   * to an independent data point, while each column corresponds to the set of
   * independent variables in a single data point.
   * @param Y An n dimensional vector of dependent variables
   * @param opts The solver options to use
   */
  GradientDescent(std::shared_ptr<ModelFunctor> &model,
                  std::shared_ptr<ModelJacobian> &jacobian,
                  const Eigen::VectorXd &A, const Eigen::MatrixXd &X,
                  const Eigen::VectorXd &Y, const SolverOpts &opts)
      : NonlinearOptimizer(model, jacobian, A, X, Y, opts) {}

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

      // Compute Current Model and Jacobians
      Eigen::VectorXd Y = Eigen::VectorXd::Zero(Y_.size());
      Eigen::MatrixXd J = Eigen::MatrixXd::Zero(X_.rows(), A_.size());
      for (int i = 0; i < J.rows(); i++) {
        // Compute Model
        Y(i) = (*mModelFunctor)(A_, X_.row(i));

        // Compute Jacobian
        J.row(i) = (*mJacobianFunctor)(A_, X_.row(i));
      }

      // Compute the gradient descent step
      Eigen::VectorXd hgd = opts_.alpha * J.transpose() * (Y_ - Y);

      // Update the model parameters
      A_ = A_ - hgd;

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