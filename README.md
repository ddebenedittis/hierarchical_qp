# Hierarchical Quadratic Programming

An implementation of the framework described in [Kinematic Control of Redundant Manipulators: Generalizing the Task-Priority Framework to Inequality Task](https://ieeexplore.ieee.org/document/5766760) and implemented in [Soft Bilinear Inverted Pendulum: A Model to Enable Locomotion With Soft Contacts](https://doi.org/10.1109/TSMC.2024.3504342).

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
[Davide De Benedittis](https://github.com/ddebenedittis)

## Citation

If you find this project useful in your research, please consider citing my related work (available [here](https://doi.org/10.1109/TSMC.2024.3504342)):

```bibtex
@ARTICLE{debenedittis2025soft,
  author={De Benedittis, Davide and Angelini, Franco and Garabini, Manolo},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems}, 
  title={Soft Bilinear Inverted Pendulum: A Model to Enable Locomotion With Soft Contacts}, 
  year={2025},
  volume={55},
  number={2},
  pages={1478-1491},
  keywords={Legged locomotion;Quadrupedal robots;Foot;Vectors;Optimization;Computational modeling;Trajectory;Tracking;Planning;Jacobian matrices;Contacts;legged locomotion;optimal control;predictive control;quadratic programming},
  doi={10.1109/TSMC.2024.3504342}}
```