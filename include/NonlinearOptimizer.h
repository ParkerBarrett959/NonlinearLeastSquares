#ifndef NONLINEAR_OPTIMIZER_H
#define NONLINEAR_OPTIMIZER_H

#include "ModelFunctor.h"
#include <Eigen/Dense>
#include <iostream>
#include <optional>

/**
 * Nonlinear Optimizer
 *
 * @brief: This is a pure virtual class defininig the interface for a nonlinear
 * optomizer.
 */
class NonlinearOptimizer {
public:
  /**
   * c'tor
   */
  NonlinearOptimizer(ModelFunctor &model) : mModelFunctor(model) {}

private:
  // underlying model function: y(x,a)
  std::shared_ptr<ModelFunctor> mModelFunctor;

  // Model weights [a1, a2, ..., an]
};

#endif // NONLINEAR_OPTIMIZER_H