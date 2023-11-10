# Hierarchical Quadratic Programming

An implementation of the framework described in [Kinematic Control of Redundant Manipulators: Generalizing the Task-Priority Framework to Inequality Task](https://ieeexplore.ieee.org/document/5766760).

## Installation

Install `numpy` and `quadprog`. Then, this can be considered as a normal ROS 2 Python package.

## Usage

```python
from hierarchical_qp.hierarchical_qp import HierarchicalQP

hqp = HierarchicalQP()

A = [...]   # list of n_tasts matrices A _of_ size (ne_i, nx)
b = [...]   # list of n_tasts vectors b of size (ne_i)
C = [...]   # list of n_tasts matrices C of size (ni_i, nx)
d = [...]   # list of n_tasts vectors d of size (ne_i)

x_star = hqp(A, b, C, d)
```

## Author

Davide De Benedittis