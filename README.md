# Hierarchical Quadratic Programming

An implementation of the framework described in [Kinematic Control of Redundant Manipulators: Generalizing the Task-Priority Framework to Inequality Task](https://ieeexplore.ieee.org/document/5766760).

## Usage

```matlab
hqp = HierarchicalQP;

A = {eye(2,4), [0,0,1,0; 1,0,0,0], eye(2,4)};
b = {ones(2,1), 10*ones(2,1), 2 * ones(2,1)};
C = {zeros(0,4), [0,0,0,1], zeros(0,4)};
d = {zeros(0,1), -12, zeros(0,1)};

x_star = hqp.solve(A, b, C, d)
```

## Author
Davide De Benedittis