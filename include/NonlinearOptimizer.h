#ifndef NONLINEAR_OPTIMIZER_H
#define NONLINEAR_OPTIMIZER_H

#include "ModelFunctor.h"
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
      numberSteps_ = 0;
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
      double yHati = (*mModelFunctor)(A_, X_(i));
      J += 0.5 * (Y_(i) - yHati) * (Y_(i) - yHati);
    }
    return J;
  }

  /**
   * Compute the gradient of the least squares cost function
   *
   * @return dJdA The gradient of the cost wrt the model parameters
   */
  Eigen::VectorXd computeGradientJ() {
    Eigen::VectorXd dJdA = Eigen::VectorXd::Zero(A_.size());
    for (int i = 0; i < Y_.size(); i++) {
      double yHati = (*mModelFunctor)(A_, X_(i));
      Eigen::VectorXd dyidA = (*mModelFunctor).gradient(A_, X_(i));
      dJdA -= (Y_(i) - yHati) * dyidA;
    }
    return dJdA;
  }

  /**
   * Is Initialized getter function
   */
  bool isInitialized() { return modelInitialized_; }

  /**
   * A getter function to check if the optimization converged
   */
  bool optimizationConverged() { return optimizerConverged_; }

  /**
   * Number of steps run by the optimizer getter function
   */
  bool getNumberSteps() { return numberSteps_; }

  /**
   * Model parameter getter function
   */
  Eigen::VectorXd getModelParameters() { return A_; }

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

  // Optimization converged flag
  bool optimizerConverged_ = false;

  // Number of iterative steps taken
  int numberSteps_;
};

#endif // NONLINEAR_OPTIMIZER_H