#ifndef GAUSS_NEWTON_H
#define GAUSS_NEWTON_H

#include "NonlinearOptimizer.h"

/**
 * Gauss-Newton Nonlinear Optimizer
 *
 * @brief: This class inherits from the NonlinearOptimizer interface and
 * implements a Gauss-Newton approach to solving the least squares problem.
 */
class GaussNewton : public NonlinearOptimizer {
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
  GaussNewton(std::shared_ptr<ModelFunctor> &model, const Eigen::VectorXd &A,
              const Eigen::VectorXd &X, const Eigen::VectorXd &Y,
              const SolverOpts &opts)
      : NonlinearOptimizer(model, A, X, Y, opts) {}

  /**
   * Function to run the Gauss-Newton optimization
   *
   * @return True if the optimization succeeded, false otherwise
   */
  bool optimize() {
    // Initial print statement
    std::cout << "Gauss-Newton optimization" << std::endl;
    std::cout << "------------------------------------------------"
              << std::endl;
    // Loop until convergence
    bool converged = false;
    int iter = 0;
    while (!converged && iter < opts_.max_iter) {
      // Increment the iteration count
      iter += 1;

      // Compute the current cost
      double J = computeJ();

      // Compute the Jacobian of the Cost Function
      Eigen::VectorXd dJdA = computeGradientJ();

      // Compute an approximate Hessian
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(A_.size(), A_.size());
      for (int i = 0; i < Y_.size(); i++) {
        Eigen::VectorXd dyidA = (*mModelFunctor).gradient(A_, X_(i));
        H += dyidA * dyidA.transpose();
      }

      // Compute the Gauss-Newton step
      Eigen::VectorXd hgn = H.inverse() * dJdA;

      // Update the model parameters
      A_ = A_ - hgn;

      // Print Current Iteration
      std::cout << "Iter = " << iter << std::endl;
      std::cout << "    Cost = " << J << std::endl;
      std::cout << "    Parameters = " << A_.transpose() << std::endl;
      std::cout << "    Step Size = " << hgn.norm() << std::endl;

      // Check for convergence
      if (hgn.norm() < opts_.convergence_criterion) {
        std::cout << "Gauss-Newton converged!" << std::endl;
        converged = true;
      }
    }
    return true;
  }
};

#endif // GAUSS_NEWTON_H