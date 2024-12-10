import numpy as np

from hierarchical_qp.hierarchical_qp import QPSolver, HierarchicalQP
from qpsolvers.exceptions import SolverNotFound


def check_array_similarity(
    a1: np.ndarray, a2: np.ndarray, threshold: float = 0.01
) -> bool:
    return np.all(np.abs(a1 - a2) <= threshold)


def test_object_creation():
    try:
        HierarchicalQP()
    except Exception as e:
        assert False, f"HierarchicalQP object raised an exception {e}."

def test_solvers():
    hqp_quadprog = HierarchicalQP(QPSolver.quadprog)
    
    A = [np.diag([1, 2, 3])]
    b = [np.array([5, 7, 11])]
    C = [None]
    d = [None]
    
    sol_quadprog = hqp_quadprog(A, b, C, d)
    
    solver_found = False
    for solver in QPSolver:
        try:
            hqp_2 = HierarchicalQP(solver)
            sol_2 = hqp_2(A, b, C, d)
        
            assert check_array_similarity(sol_quadprog, sol_2), \
                'The solutions with only equality constraints between the ' \
                f'{QPSolver.quadprog} and {solver} solvers differ.'
                
            solver_found = True
        except (SolverNotFound, ValueError):
            pass
        
    assert solver_found, 'No QP solver was found.'
            
    # A = [np.diag([0, 0, 0]), None]
    # b = [np.array([0, 0, 0]), None]
    # C = [None, np.eye(3)]
    # d = [None, np.array([-1, -2, -3])]
    
    # sol_quadprog = hqp_quadprog(A, b, C, d)
    
    # for solver in QPSolver:
    #     hqp_2 = HierarchicalQP(solver)
    #     sol_2 = hqp_2(A, b, C, d)
    
    #     assert check_array_similarity(sol_quadprog, sol_2), \
    #         'The solutions with equality and inequality constraints between ' \
    #         f'the {QPSolver.quadprog} and {solver} solvers differ.'
