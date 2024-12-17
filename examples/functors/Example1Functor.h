#ifndef EXAMPLE_1_FUNCTOR_H
#define EXAMPLE_1_FUNCTOR_H

#include "ModelFunctor.h"
#include <Eigen/Dense>

/**
 * Example 1 functor
 *
 * @brief: This functor implements the linear function: y(a,x) = a1*x + a2*x^2 +
 * a3*x^3. Notez: This is linear in the parameters, [a1, a2, a3], only.
 */
class Example1Functor : public ModelFunctor {
public:
  /**
   * Functor operator
   *
   * @param a A vector containing the model parameters that are being solved for
   * @param x A double representing the current independent variables to
   * evaluate the model at
   * @return A double representing the model output value
   */
  double operator()(const Eigen::VectorXd &a, const double x) override {
    double x10 = x / 10.0;
    return (a(0) * x10 + a(1) * (x10*x10) + a(2) * (x10*x10*x10) + a(3) * (x10*x10*x10*x10));
  }

  /**
   * Gradient of the functor
   *
   * @param a A vector containing the model parameters that are being solved for
   * @param x A double representing the current independent variables to
   * evaluate the model at
   * @return A vector representing the gradient of the model wrt the parameters
   */
  Eigen::VectorXd gradient(const Eigen::VectorXd &a, const double x) override {
    double x10 = x / 10.0;
    Eigen::VectorXd dyda(a.size());
    dyda << x10, x10*x10, x10*x10*x10, x10*x10*x10*x10;
    return dyda;
  }
};
#endif // EXAMPLE_1_FUNCTOR_H
