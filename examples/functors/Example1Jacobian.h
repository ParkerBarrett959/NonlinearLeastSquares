#ifndef EXAMPLE_1_JACOBIAN_H
#define EXAMPLE_1_JACOBIAN_H

#include "ModelJacobian.h"

/**
 * Example 1 Jacobians
 *
 * @brief: This Jacobian implements the Jacobian of the linear Jacobian above.
 */
class Example1Jacobian : public ModelJacobian {
public:
  /**
   * Jacobian operator
   *
   * @param a A vector containing the model parameters that are being solved for
   * @param x A vector containing the current independent variables to evaluate
   * @param e A double containing the current error, e = y_hat - yi
   * the model at
   * @return A 1x3 Jacobian matrix of the function above
   */
  Eigen::MatrixXd operator()(const Eigen::VectorXd &a, const Eigen::VectorXd &x, const double e) override {
    // Assertions - Use these to prevent errors due to passing incorrectly sized values
    assert(a.size() == 3);
    assert(x.size() == 1);

    // Calculate Jacobian terms
    double dyda1 = e * x(0);
    double dyda2 = e * x(0)*x(0);
    double dyda3 = e * x(0)*x(0)*x(0);
    
    // Create Jacobian Matrix
    Eigen::Matrix<double, 1, 3> J;
    J << dyda1, dyda2, dyda3;
    return J;
  }
};

#endif // EXAMPLE_1_Jacobian_H