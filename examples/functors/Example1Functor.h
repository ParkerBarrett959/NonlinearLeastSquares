#ifndef EXAMPLE_1_FUNCTOR_H
#define EXAMPLE_1_FUNCTOR_H

#include "ModelFunctor.h"
#include <Eigen/Dense>

/**
 * Example 1 functor
 *
 * @brief: This functor implements the linear function: y(a,x) = a1*x1 + a2*x2^2 + a3*x3^2.
 * Notez: This is linear in the parameters, [a1, a2, a3], only. 
 */
class Example1Functor : public ModelFunctor {
public:
  /**
   * Functor operator
   *
   * @param a A vector containing the model parameters that are being solved for
   * @param x A vector containing the current independent variables to evaluate
   * the model at
   * @return A double representing the model output value
   */
  double operator()(const Eigen::VectorXd &a, const Eigen::VectorXd &x) override {
    // Assertions - Use these to prevent errors due to passing incorrectly sized values
    assert(a.size() == 3);
    assert(x.size() == 3);

    // Evaluate function
    return (a(0)*x(0) + a(1)*x(1)*x(1) + a(2)*x(2)*x(2)*x(2));
  }
};
#endif // EXAMPLE_1_FUNCTOR_H