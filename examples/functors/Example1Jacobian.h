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
   * the model at
   * @return A 1x3 Jacobian matrix of the function above
   */
  Eigen::MatrixXd operator()(const Eigen::VectorXd &a, const Eigen::VectorXd &x) override {
    // Assertions - Use these to prevent errors due to passing incorrectly sized values
    assert(a.size() == 3);
    assert(x.size() == 3);

    // Calculate Jacobian terms
    double dydx1 = a(0);
    double dydx2 = 2*a(1)*x(1);
    double dydx3 = 3*a(2)*x(2)*x(2);
    
    // Create Jacobian Matrix
    Eigen::Matrix<double, 1, 3> J;
    J << dydx1, dydx2, dydx3;
    return J;
  }
};

#endif // EXAMPLE_1_Jacobian_H