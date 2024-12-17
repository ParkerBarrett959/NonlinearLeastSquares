#ifndef NONLINEAR_OPTIMIZER_H
#define NONLINEAR_OPTIMIZER_H

#include "ModelFunctor.h"
#include "ModelJacobian.h"
#include "SolverOpts.h"
#include <iostream>
#include <memory>

/**
 * Nonlinear Optimizer
 *
 * @brief: This is a pure virtual class defininig the interface for a nonlinear
 * optimizer.
 */
class NonlinearOptimizer {
public:
  /**
   * c'tor
   */
  NonlinearOptimizer(std::shared_ptr<ModelFunctor> &model,
                     const Eigen::VectorXd &A, const Eigen::VectorXd &X,
                     const Eigen::VectorXd &Y, const SolverOpts &opts)
      : mModelFunctor(model), A_(A), X_(X), Y_(Y), opts_(opts) {
    // Verify X and Y dimensions are correct
    if (X_.size() == Y_.size()) {
      modelInitialized_ = true;
    }
  }

  /**
   * Pure virtual function to solve the model
   */
  virtual bool optimize() = 0;

  /**
   * Compute the least squares cost function
   * 
   * @return J The sum of the squares of the residuals
   */
  double computeJ() {
    double J = 0.0;
    for (int i = 0; i < Y_.size(); i++) {
      double yHati = (*mModelFunctor)(A_, X_.row(i));
      J += (Y_(i) - yHati) * (Y_(i) - yHati)
    }
    return J;
  }


  /**
   * Is Initialized getter function
   */
  bool isInitialized() { return modelInitialized_; }

protected:
  // underlying model function: y(x,A)
  std::shared_ptr<ModelFunctor> mModelFunctor;

  // Vector of model weights [a1, a2, ..., an]
  Eigen::VectorXd A_;

  // Vector of independent variables [x1, x2, ..., xm]
  Eigen::VectorXd X_;

  // Vector of dependent variables [y1, ..., ym]
  Eigen::VectorXd Y_;

  // Solver Options
  SolverOpts opts_;

  // Model intialized flag
  bool modelInitialized_ = false;
};

#endif // NONLINEAR_OPTIMIZER_H