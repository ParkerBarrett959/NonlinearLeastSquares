#ifndef SOLVER_OPTS_H
#define SOLVER_OPTS_H

/**
 * Solver options
 *
 * @brief: A struct containing options for configuring the nonlinear least
 * squares solvers.
 */
struct SolverOpts {
  // Maximum number of iterations
  int max_iter = 1000;

  // Convergence criterion (change in parameters between steps)
  double convergence_criterion = 1e-3;
};

#endif // SOLVER_OPTS_H