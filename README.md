# Hierarchical Quadratic Programming

An implementation of the framework described in [Kinematic Control of Redundant Manipulators: Generalizing the Task-Priority Framework to Inequality Task](https://ieeexplore.ieee.org/document/5766760).

## Usage

For HierarchicalQP.m
```matlab
hqp = HierarchicalQP;

A = {eye(2,4), [0,0,1,0; 1,0,0,0], eye(2,4)};
b = {ones(2,1), 10*ones(2,1), 2 * ones(2,1)};
C = {zeros(0,4), [0,0,0,1], zeros(0,4)};
d = {zeros(0,1), -12, zeros(0,1)};

x_star = hqp.solve(A, b, C, d)
```

For HierarchicalQP_solver.m
```matlab
A = {eye(2,4), [0,0,1,0; 1,0,0,0], eye(2,4)};
b = {ones(2,1), 10*ones(2,1), 2 * ones(2,1)};
C = {zeros(0,4), [0,0,0,1], zeros(0,4)};
d = {zeros(0,1), -12, zeros(0,1)};
x_init = zeros(4,1);

x_star = HierarchicalQP_solver(A_hqp, b_hqp, C_hqp, d_hqp,{},{},{},x_init);

```
For HierarchicalQP_solver_nocell.m
```matlab
A = [eye(2,4); [0,0,1,0; 1,0,0,0]; eye(2,4)];
b = [ones(2,1); 10*ones(2,1); 2 * ones(2,1)];
C = [zeros(0,4); [0,0,0,1]; zeros(0,4)];
d = [zeros(0,1); -12; zeros(0,1)];
dimarray_eq = [2 2 2];
dimarray_ineq = [0 1 0];
priority = [1 2 3];
x_init = zeros(4,1);

x_star = HierarchicalQP_solver_nocell(A, b, C, d, dimarray_eq ,dimarray_ineq, priority,x_init);
```

## Author
Davide De Benedittis

Modified by Fanyi Kong
