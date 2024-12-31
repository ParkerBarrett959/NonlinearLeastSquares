#ifndef EXAMPLE_2_FUNCTOR_H
#define EXAMPLE_2_FUNCTOR_H

#include "ModelFunctor.h"
#include <Eigen/Dense>
#include <cmath>

/**
 * Example 2 functor
 *
 * @brief: This functor implements the nonlinear function: y(a,x) =
 * a1*exp(-x/a2) + a3*x*exp(-x/a4)
 */
class Example2Functor : public ModelFunctor {
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
    return (a(0) * std::exp(-x / a(1)) + a(2) * x * std::exp(-x / a(3)));
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
    Eigen::VectorXd dyda(a.size());
    dyda << std::exp(-x / a(1)),
        a(0) * std::exp(-x / a(1)) * (x / (a(1) * a(1))),
        x * std::exp(-x / a(3)),
        a(2) * std::exp(-x / a(3)) * (x * x / (a(3) * a(3)));
    return dyda;
  }
};
#endif // EXAMPLE_2_FUNCTOR_H
