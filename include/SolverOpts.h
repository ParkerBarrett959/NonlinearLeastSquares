#ifndef SOLVER_OPTS_H
#define SOLVER_OPTS_H

/**
 * Solver options
 *
 * @brief: A struct containing options for configuring the nonlinear least
 * squares solvers.
 */
struct SolverOpts {
  // General Options
  int max_iter = 100;                  // max number of iterations
  double convergence_criterion = 1e-3; // convergence criterion

  // Gradient Descent options
  double alpha = 0.01; // learning rate
};

#endif // SOLVER_OPTS_H