#ifndef MODEL_JACOBIAN_H
#define MODEL_JACOBIAN_H

#include <Eigen/Dense>
#include <cassert>

/**
 * Model Jacobian
 *
 * @brief: This is a pure virtual class defining the interface for model
 * Jacobians. Derive from this class to create your own models to solve
 * nonlinear least squares problems.
 */
class ModelJacobian {
public:
  /**
   * Jacobian operator
   *
   * @brief: This function is overriden in the inherited class and implements
   * the model. Note: If the incorrect sized a or x vectors are passed, the
   * model may throw an exception at runtime.
   *
   * @param a A vector containing the model parameters that are being solved for
   * @param x A vector containing the current independent variables to evaluate
   * the model at
   * @return An Eigen::MatrixXd representing the model output value
   */
  virtual Eigen::MatrixXd operator()(const Eigen::VectorXd &a,
                                     const Eigen::VectorXd &x) = 0;
};

#endif // MODEL_Jacobian_H