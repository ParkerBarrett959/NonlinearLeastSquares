#ifndef NONLINEAR_OPTIMIZER_H
#define NONLINEAR_OPTIMIZER_H

#include <Eigen/Dense>
#include <iostream>
#include <optional>

/**
 * Nonlinear Optimizer
 *
 * @brief: This is a pure virtual class defininig the interface for a nonlinear optomizer.
 */
class NonlinearOptimizer {
public:
  /**
   * c'tor
   */
  NonlinearOptimizer();

  

private:
  // underlying model function: y(x,a)

  // Model weights [a1, a2, ..., an]
};

#endif // NONLINEAR_OPTIMIZER_H