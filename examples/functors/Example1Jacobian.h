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
    double dyda1 = x(0);
    double dyda2 = x(1)*x(1);
    double dyda3 = x(2)*x(2)*x(2);
    
    // Create Jacobian Matrix
    Eigen::Matrix<double, 1, 3> J;
    J << dyda1, dyda2, dyda3;
    return J;
  }
};

#endif // EXAMPLE_1_Jacobian_H