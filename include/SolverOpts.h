#ifndef SOLVER_OPTS_H
#define SOLVER_OPTS_H

/**
 * Solver options
 *
 * @brief: A struct containing options for configuring the nonlinear least
 * squares solvers. Since the same struct is used for all solvers, solver
 * specific options must still be set regardless of the solver used. Changing
 * the parameters will have no effect on the solution however if you are not
 * using the particular solver to which the parameter pertains.
 */
struct SolverOpts {
  // General Options
  int max_iter = 1000;                   // max number of iterations
  double convergence_criterion = 1.0e-3; // convergence criterion
  bool print_steps = false;              // print each step

  // Gradient Descent Specific Options
  double alpha = 0.1; // learning rate

  // Levenberg-Marquardt Specific Options
  double lambda0 = 1.0; // Initial damping coefficient
  double factor = 2.0;  // Factor to scale damping coefficient by
};

#endif // SOLVER_OPTS_H