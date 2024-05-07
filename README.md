# Hierarchical Quadratic Programming

An implementation of the framework described in [Kinematic Control of Redundant Manipulators: Generalizing the Task-Priority Framework to Inequality Task](https://ieeexplore.ieee.org/document/5766760).

## Installation

Install the Python requirements with
```shell
pip install -r requirements.txt
```

## Usage

```python
from hierarchical_qp.hierarchical_qp import HierarchicalQP, QPSolver

solver = QPSolver.quadprog  # or QPSolver.osqp
hqp = HierarchicalQP(solver=solver, hierarchical=True)

A = [...]   # list of n_tasts matrices A _of_ size (ne_i, nx)
b = [...]   # list of n_tasts vectors b of size (ne_i)
C = [...]   # list of n_tasts matrices C of size (ni_i, nx)
d = [...]   # list of n_tasts vectors d of size (ne_i)

x_star = hqp(A, b, C, d)
```

If a task has no equality or inequality part at a certain priority, you can use `None`.

The first task must have an equality constraint part.

## Author

Davide De Benedittis