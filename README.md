# Hierarchical Quadratic Programming

An implementation of the framework described in [Kinematic Control of Redundant Manipulators: Generalizing the Task-Priority Framework to Inequality Task](https://ieeexplore.ieee.org/document/5766760).

## Installation

Just run the following command without downloading the repo to install it:
```
pip3 install git+https://github.com/ddebenedittis/hierarchical_qp.git
```

## Usage

```python
from hierarchical_qp import HierarchicalQP

hqp = HierarchicalQP()

A = [...]   # list of n_tasts matrices A _of_ size (ne_i, nx)
b = [...]   # list of n_tasts vectors b of size (ne_i)
C = [...]   # list of n_tasts matrices C of size (ni_i, nx)
d = [...]   # list of n_tasts vectors d of size (ne_i)

x_star = hqp(A, b, C, d)
```

If b[i] or d[i] are matrices, they are converted to vectors.

## Author
Davide De Benedittis