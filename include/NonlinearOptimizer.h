#ifndef NONLINEAR_OPTIMIZER_H
#define NONLINEAR_OPTIMIZER_H

#include "ModelFunctor.h"
#include <Eigen/Dense>
#include <iostream>
#include <memory>

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
  NonlinearOptimizer(std::shared_ptr<ModelFunctor> &model,
                     const Eigen::VectorXd &A, const Eigen::MatrixXd &X,
                     const Eigen::VectorXd &Y)
      : mModelFunctor(model), A_(A), X_(X), Y_(Y) {
    // Verify X and Y dimensions are correct
    if (X_.rows() == Y_.size()) {
      modelInitialized_ = true;
    }
  }

  /**
   * Pure virtual function to solve the model
   */
  virtual bool optimize() = 0;

protected:
  // underlying model function: y(x,a)
  std::shared_ptr<ModelFunctor> mModelFunctor;

  // Vector of model weights [a1, a2, ..., ap]
  Eigen::VectorXd A_;

  // Matrix of independent variables (nxm matrix) with each row corresponding to
  // a sample and each column corresponding to [x1, ..., xm])
  Eigen::MatrixXd X_;

  // Vector of dependent variables [y1, ..., yn]
  Eigen::VectorXd Y_;

  // Model intialized flag
  bool modelInitialized_ = false;
};

#endif // NONLINEAR_OPTIMIZER_H