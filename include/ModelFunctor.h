#ifndef MODEL_FUNCTOR_H
#define MODEL_FUNCTOR_H

#include <Eigen/Dense>
#include <cassert>

/**
 * Model functor
 *
 * @brief: This is a pure virtual class defining the interface for model
 * functors. Derive from this class to create your own models to solve nonlinear
 * least squares problems.
 */
class ModelFunctor {
public:
  /**
   * Functor operator
   *
   * @brief: This function is overriden in the inherited class and implements
   * the model. Note: If the incorrect sized A vector is passed, the model may
   * throw an exception at runtime.
   *
   * @param A A vector containing the model parameters that are being solved for
   * @param x A double representing the current independent variables to
   * evaluate the model at
   * @return A double representing the model dependent variable, y
   */
  virtual double operator()(const Eigen::VectorXd &A, const double x) = 0;

  /**
   * Gradient of the functor
   *
   * @brief: This function is overriden in the inherited class and implements
   * the gradient of the model. Note: If the incorrect sized A vector is passed,
   * the model may throw an exception at runtime.
   *
   * @param A A vector containing the model parameters that are being solved for
   * @param x A double representing the current independent variables to
   * evaluate the model at
   * @return A vector containing the gradient of the function
   */
  virtual Eigen::VectorXd gradient(const Eigen::VectorXd &A,
                                   const double x) = 0;
};

#endif // MODEL_FUNCTOR_H