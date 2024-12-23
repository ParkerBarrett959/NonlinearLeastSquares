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
  int max_iter = 1000;                   // max number of iterations
  double convergence_criterion = 1.0e-3; // convergence criterion
  double alpha = 0.1;                    // learning rate
  bool print_steps = false;              // print each step
};

#endif // SOLVER_OPTS_H