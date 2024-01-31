from enum import auto, Enum
import numpy as np
from scipy.sparse import csc_matrix

import osqp
from quadprog import solve_qp



class QPSolver(Enum):
    """QP solver type."""
    quadprog = auto()
    osqp = auto()


def null_space_projector(A):
    """
    Compute the null space projector using the Moore-Penrose pseudo-inverse.
    """
    
    N = np.eye(A.shape[1]) - np.linalg.pinv(A) @ A

    return N



# ============================================================================ #
#               HIERARCHICAL QUADRATIC PROGRAMMING IMPLEMENTATION              #
# ============================================================================ #

# A general task T can be defined as
#       [ we * (A x - b)  = v
#   T = [
#       [ wi * (C x - d) <= w

# where v and w are slack variables.

# Is is formulated as a QP problem
#   min_x 1/2 (A x - b)^2 + 1/2 w^2
#   s.t.: C x - d <= w


# It can be rewritten in the general QP form:
#   min_x 1/2 xi^T H xi + p^T xi
#   s.t: CI xi + ci0 >= 0

# where:
#   H   =   A^T A
#   p   = - A^T b
#   CI  = [ -C, 0 ]
#         [  0, I ]
#   ci0 = [ d ]
#         [ 0 ]
#   xi  = [ x ]
#         [ w ]

# ============================================================================

# Given a set of tasks T1, ..., Tn, for the task Tp the QP problem becomes:
#   H   = [ Zq^T Ap^T Ap Zq, 0 ]
#         [               0, I ]
#   p   = [ Zq^T Ap^T (Ap x_opt - bp) ]
#         [                         0 ]

#   CI  = [   0,       I      ]
#         [ - C_stack, [0; I] ]
#   ci0 = [ 0                                    ]
#         [ d - C_stack x_opt + [w_opt_stack; 0] ]

# The solution of the task with priority p is x_p_star

# The solution of the tasks with priority equal or smaller that p+1 is
# x_p+1_star_bar = x_p_star_bar + Z @ x_p+1_star


# ============================================================================ #

class HierarchicalQP:
    def __init__(self, solver: QPSolver = QPSolver.quadprog):
        # Small number used to make H positive definite.
        self._regularization = 1e-6
        
        self.solver = solver
        
        self.osqp_solvers = []
        
    @property
    def regularization(self):
        return self._regularization
    
    @regularization.setter
    def regularization(self, value):
        try:
            if float(value) < 0:
                raise ValueError()
                
            self._regularization = float(value)
        except ValueError:
            raise ValueError('"regularization" must be a positive number') from None
        
    
    @staticmethod
    def check_dimensions(A, b, C, d, we, wi, priorities):
        """
        Raise ValueError if the dimension of the input matrices are not consistent.
        Additonally, None or empty matrices are converted into empty matrices of
        opportune size.
        """
        
        n_tasks = len(A)
        
        nx = A[0].shape[1]
        
        # Convert empty matrices or None into empty matrices of opportune size.
        for i in range(n_tasks):
            if A[i] is None or A[i].size == 0:
                A[i] = np.empty([0,nx])
                
            if b[i] is None or b[i].size == 0:
                b[i] = np.empty([0])
            else:
                b[i] = b[i].flatten()
            
            if C[i] is None or C[i].size == 0:
                C[i] = np.empty([0,nx])
                
            if d[i] is None or d[i].size == 0:
                d[i] = np.empty([0])
            else:
                d[i] = d[i].flatten()
                
        # Chech that the priorities list is correctly constructed
        if priorities is not None:
            for i in range(n_tasks):
                if i not in priorities:
                    raise ValueError(
                        f"priorities is ill formed: priorities = {priorities}"
                    )
        
        if len(b) != n_tasks \
                or len(C) != n_tasks \
                or len(d) != n_tasks \
                or (priorities is not None and len(priorities) != n_tasks):
            raise ValueError(
                "A, b, C, d, priorities must be lists of the same length." + \
                f"Received lists of {len(A)}, {len(b)}, {len(C)}, " + \
                f"{len(d)}, {len(priorities)} elements."
            )
            
        for i in range(n_tasks):
            if A[i].shape[0] != b[i].shape[0]:
                raise ValueError(
                    f"At priority {i}, A and b have a different number of rows."
                )
                
            if C[i].shape[0] != d[i].shape[0]:
                raise ValueError(
                    f"At priority {i}, C and d have a different number of rows."
                )

        for i in range(n_tasks):
            if A[i].shape[1] != nx:
                raise ValueError(
                    f"At priority {i}, A has {A[i].shape[1]} columns instead of {nx}"
                )
                
            if C[i].shape[1] != nx:
                raise ValueError(
                    f"At priority {i}, C has {C[i].shape[1]} columns instead of {nx}"
                )
                
        if we is not None:
            for p, we_p in enumerate(we):
                if we_p is not None and  we_p.size != A[p].shape[0]:
                    raise ValueError(
                        f"At priority {p}, we has {we_p.size} elements " + \
                        f"instead of {A[p].shape[0]}"
                    )
                    
        if wi is not None:
            for p, wi_p in enumerate(wi):
                if wi_p is not None and  wi_p.size != C[p].shape[0]:
                    raise ValueError(
                        f"At priority {p}, wi has {wi_p.size} elements " + \
                        f"instead of {C[p].shape[0]}"
                    )
            

    def __call__(
        self, A, b, C, d, we = None, wi = None, priorities = None
    ) -> np.ndarray:
        """
        Given a set of tasks in the form \\
        Ap x  = b \\
        Cp x <= d, \\
        with p = 1:p_max, return the optimal vector x_star that solves the
        hierarchical QP problem.

        Args:
            A (list[np.ndarray]): list of Ap matrices of size (ne_p, nx)
            b (list[np.ndarray]): list of bp vectors of size (ne_p)
            C (list[np.ndarray]): list of Cp matrices of size (ni_p, nx)
            d (list[np.ndarray]): list of dp vectors of size (ni_p)
            we (list[np.ndarray]): list of we_p vectors of size (ne_p)
            wi (list[np.ndarray]): list of wi_p vectors of size (ni_p)
            priorities (list[int]): list of ints representing the priorities of
                                    the tasks, from 0 to p_max - 1

        Returns:
            np.ndarray: optimal solution vector
        """
        
        
        # ========================== Initialization ========================== #

        # Number of tasks.
        n_tasks = len(A)

        # Dimension of the optimization vector.
        nx = A[0].shape[1]

        # Optimization vector.
        x_star_bar = np.zeros(nx)
        
        # History of the slack variables, stored as a list of np.arrays.
        w_star_bar = [np.empty(shape = [0,])]   

        # Initialize the null space projector.
        Z = np.eye(nx)
        
        
        self.check_dimensions(A, b, C, d, we, wi, priorities)
        
        if len(A) != len(self.osqp_solvers):
            self.osqp_solvers = [osqp.OSQP() for i in range(len(A))]

        # ==================================================================== #

        for i in range(n_tasks):
            # Priority of task i.
            if priorities is None:
                priority = i
            else:
                priority = priorities.index(i)
            
            Ap = A[priority]
            bp = b[priority]
            
            if we is not None:
                if we[priority] is not None:
                    Ap = Ap * we[priority][:, np.newaxis]
                    bp = bp * we[priority]
                    
            if wi is not None:
                if wi[priority] is not None:
                    C[priority] = C[priority] * wi[priority][:, np.newaxis]
                    d[priority] = d[priority] * wi[priority]
            
            # Slack variable dimension at task p.
            nw = C[priority].shape[0]
            
            
            # See Kinematic Control of Redundant Manipulators: Generalizing the
            # Task-Priority Framework to Inequality Task for the math behind it.

            # ======================== Compute H And P ======================= #

            if Ap.size != 0:
                H = np.block([
                    [Z.T @ Ap.T @ Ap @ Z,  np.zeros([nx,nw])],
                    [  np.zeros([nw,nx]),         np.eye(nw)],
                ])

                p = np.block([
                    Z.T @ Ap.T @ (Ap @ x_star_bar - bp),
                    np.zeros(nw)
                ])
            else:
                H = np.block([
                    [np.zeros([nx,nx]),  np.zeros([nx,nw])],
                    [np.zeros([nw,nx]),         np.eye(nw)],
                ])

                p = np.zeros(nx+nw)
                
            # Make H positive definite
            H = H + self._regularization * np.eye(H.shape[0])
            
            # ================== Compute C_tilde And D_tilde ================= #

            nC2 = np.concatenate(C[0:priority+1]).shape[0]

            C_tilde = np.block([
                [                  np.zeros([nw,nx]),         - np.eye(nw)],
                [np.concatenate(C[0:priority+1]) @ Z,   np.zeros([nC2,nw])],
            ])
            if nw > 0:
                C_tilde[-nw:, -nw:] = - np.eye(nw)

            # w_star_arr = [w_star[priority], w_star[priority-1], ..., w_star[0]]
            w_star_arr = np.concatenate(w_star_bar[:])

            d_tilde = np.block([
                np.zeros(nw),
                np.concatenate(d[0:priority+1]) \
                    - np.concatenate(C[0:priority+1]) @ x_star_bar \
                    + np.concatenate((w_star_arr, np.zeros(nw))),
            ])
            d_tilde = d_tilde.flatten()


            # =========================== Solve The QP =========================== #
            
            # Quadprog library QP problem formulation
            #   min  1/2 x^T H x - p^T x
            #   s.t. CI^T x >= ci0

            # The quadprog library defines some matrices differently from here.
            if C_tilde.size == 0:
                if self.solver == QPSolver.quadprog:
                    sol = solve_qp(H, -p)[0]
                elif self.solver == QPSolver.osqp:
                    self.osqp_solvers[i].setup(
                        csc_matrix(H), p,
                        None, None, None,
                        verbose=False
                    )
                    ret = self.osqp_solvers[i].solve()
                    sol = ret.x
            else:
                if self.solver == QPSolver.quadprog:
                    try:
                        sol = solve_qp(H, -p, -C_tilde.T, -d_tilde)[0]
                    except ValueError as e:
                        print(f"At priority {priority}: " + str(e))
                        return x_star_bar
                    
                elif self.solver == QPSolver.osqp:
                    self.osqp_solvers[i].setup(
                        csc_matrix(H), p,
                        csc_matrix(C_tilde), None, d_tilde,
                        verbose=False
                    )
                    ret = self.osqp_solvers[i].solve()
                    sol = ret.x
                        

            # ======================== Post-processing ======================= #

            # Extract x_star from the solution.
            x_star = sol[0:nx]
            
            # Update the solution of all the tasks up to now.
            x_star_bar = x_star_bar + Z @ x_star

            # Store the history of w_star
            if priority == 0:
                w_star_bar = [sol[nx:]]
            else:
                w_star_bar.append(sol[nx:])

            # Compute the new null space projector (skipped at the last iteration).
            if ((Ap.shape[0] != 0) and (priority != n_tasks - 1)):
                Z = Z @ null_space_projector(Ap @ Z)
                
            # End the loop if Z is the null matrix.
            if not np.any((Z > self.regularization) | (Z < -self.regularization)):
                return x_star_bar

        return x_star_bar
