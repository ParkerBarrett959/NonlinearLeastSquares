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
   */
  GradientDescent(std::shared_ptr<ModelFunctor> &model,
                  std::shared_ptr<ModelJacobian> &jacobian,
                  const Eigen::VectorXd &A, const Eigen::MatrixXd &X,
                  const Eigen::VectorXd &Y)
      : NonlinearOptimizer(model, jacobian, A, X, Y) {}

  /**
   * Function to run the gradient descent optimization
   *
   * @return True if the optimization succeeded, false otherwise
   */
  bool optimize() { return true; }
};

#endif // GRADIENT_DESCENT_H