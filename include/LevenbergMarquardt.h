#ifndef LEVENBERG_MARQUARDT_H
#define LEVENBERG_MARQUARDT_H

#include "NonlinearOptimizer.h"

/**
 * Levenberg-Marquardt Nonlinear Optimizer
 *
 * @brief: This class inherits from the NonlinearOptimizer interface and
 * implements a Levenberg-Marquardt approach to solving the least squares
 * problem.
 */
class LevenbergMarquardt : public NonlinearOptimizer {
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
  LevenbergMarquardt(std::shared_ptr<ModelFunctor> &model,
                     const Eigen::VectorXd &A, const Eigen::VectorXd &X,
                     const Eigen::VectorXd &Y, const SolverOpts &opts)
      : NonlinearOptimizer(model, A, X, Y, opts) {}

  /**
   * Function to run the Levenberg-Marquardt optimization
   *
   * @return True if the optimization succeeded, false otherwise
   */
  bool optimize() {
    // Initial print statement
    std::cout << "Running Levenberg-Marquardt optimization..." << std::endl;

    // Set damping coefficient
    double lambda = opts_.lambda0;

    // Loop until convergence
    while (!optimizerConverged_ && numberSteps_ <= opts_.max_iter) {
      // Increment the iteration count
      numberSteps_ += 1;

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

      // Compute the Levenberg-Marquardt step with the current value of lambda
      // and a smaller value
      Eigen::MatrixXd I = Eigen::MatrixXd::Identity(A_.size(), A_.size());
      Eigen::VectorXd hlm_lambdaCurr = (H + lambda * I).inverse() * dJdA;
      Eigen::VectorXd hlm_lambdaSmaller =
          (H + (lambda / opts_.factor) * I).inverse() * dJdA;

      // Compute the cost function at each of the Levenberg-Marquardt steps
      double JLambdaCurr = computeJ(A_ - hlm_lambdaCurr);
      double JLambdaSmaller = computeJ(A_ - hlm_lambdaSmaller);

      // Check for both new costs being worse. In this case, lambda should be
      // increased which will have the effect of damping (decreasing) the step.
      if (JLambdaCurr > J && JLambdaSmaller > J) {
        lambda *= opts_.factor;
        continue;
      }

      // If the smaller lambda value reduced the cost, this value is used and
      // the corresponding step is applied. Smaller values of lambda trend the
      // solver towards Gauss-Newton which performs better near the equilibrium.
      // If the larger lambda was the only improvement, this will be used
      // instead. Larger values of lambda trend the problem closer towards
      // Gradient Descent.
      Eigen::VectorXd hlm = Eigen::VectorXd::Zero(A_.size());
      if (JLambdaSmaller <= J) {
        lambda = lambda / opts_.factor;
        hlm = hlm_lambdaSmaller;
      } else {
        hlm = hlm_lambdaCurr;
      }

      // Update the model parameters
      A_ = A_ - hlm;

      // Check for convergence and print step
      if (hlm.norm() < opts_.convergence_criterion) {
        if (opts_.print_steps) {
          std::cout << "Levenberg-Marquardt converged!" << std::endl;
          std::cout << "Number of Iterations: " << numberSteps_ << std::endl;
          std::cout << "Cost: " << J << std::endl;
          std::cout << "Final Step Size: " << hlm.norm() << std::endl;
          std::cout << "Final Model Parameters: " << A_.transpose()
                    << std::endl;
        }
        optimizerConverged_ = true;
      } else {
        if (opts_.print_steps) {
          std::cout << "i = " << numberSteps_ << ", J = " << J
                    << ", step = " << hlm.norm() << std::endl;
        }
      }
    }
    std::cout << "Levenber-Marquardt Complete!" << std::endl;
    return true;
  }
};

#endif // LEVENBERG_MARQUARDT_H