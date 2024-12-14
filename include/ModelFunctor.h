#ifndef MODEL_FUNCTOR_H
#define MODEL_FUNCTOR_H

#include <Eigen/Dense>

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
   * the model. Note: If the incorrect sized a or x vectors are passed, the
   * model may throw an exception at runtime.
   *
   * @param a A vector containing the model parameters that are being solved for
   * @param x A vector containing the current independent variables to evaluate
   * the model at
   * @return A double representing the model output value
   */
  virtual double operator()(const Eigen::VectorXd &a,
                            const Eigen::VectorXd &x) = 0;
};

#endif // MODEL_FUNCTOR_H