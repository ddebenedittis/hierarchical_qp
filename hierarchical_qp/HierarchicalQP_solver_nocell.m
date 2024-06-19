% No cell function version, an example for use:
% x_km1 = zeros(4,1);
% A_eq1 = [1 1 1 1; 2 2 2 2];
% b_eq1 = [1;2];
% 
% D_eq1 = [3 3 3 3; 4 4 4 4];
% f_eq1 = [3;4];
% 
% A_eq2 = [5 5 5 5];
% b_eq2 = [5];
% 
% D_eq2 = [6 6 6 6];
% f_eq2 = [6];
% 
% A_hqp = [A_eq1;A_eq2];
% b_hqp = [b_eq1;b_eq2];
% D_hqp = [D_eq1;D_eq2];
% f_hqp = [f_eq1;f_eq2];
% 
% dimarray_eq = [2 0 1 0];
% dimarray_ineq = [0 2 0 1];
% priority = [1 2 3 4];
% x_star = HierarchicalQP_solver_nocell(A_hqp, b_hqp, D_hqp, f_hqp, dimarray_eq ,dimarray_ineq, priority,x_km1);

function  x_star = HierarchicalQP_solver_nocell(A, b, D, f, dimarray_eq, dimarray_ineq, priorities, x_init)
    
    regularization = 1e-6;
    % ======================== Check Dimensions ======================== %
    if sum(dimarray_eq) ~= size(A, 1) || sum(dimarray_eq) ~= size(b, 1)
        error("Equality Constraint dimensions don't match")
    end

    if sum(dimarray_ineq) ~= size(D, 1) || sum(dimarray_ineq) ~= size(f, 1)
        error("Inequality Constraint dimensions don't match")
    end

    if size(x_init,1) ~= size(A, 2) || size(x_init,1) ~= size(D, 2)
        error("Variable dimension don't match")
    end
    
    if isempty(priorities)
        priorities = 1:length(dimarray_eq);
    else

    if length(dimarray_eq) ~= length(priorities) || length(dimarray_ineq) ~= length(priorities)
        error("Task number don't match")
    end
    % ======================== Initialization ======================== %
    n_tasks = length(dimarray_eq);

    % Dimension of the optimization vector.
    n_x = size(A, 2);

    % Optimization vector.
    x_star = zeros(n_x, 1);

    % History of the slack variables, stored as a list of np.arrays.
    coder.varsize('D_cat','f_cat','v_cat')
    D_cat = zeros(0,n_x);
    f_cat =  zeros(0,1);
    v_cat = zeros(0,1); 
    % Initialize the null space projector.
    Z = eye(n_x);

    index_arr = zeros(1,length(priorities));
    % ================================================================ %
    for p = 1:n_tasks
        count = p;
        index = find(priorities == p);
        index_arr(p) = index;

        if 0 == dimarray_eq(index) 
            A_p = zeros(0,n_x);
            b_p = zeros(0,1);
        else    
            A_p = A(sum(dimarray_eq(index_arr(1:p))) - dimarray_eq(index)+1:sum(dimarray_eq(index_arr(1:p))),:);
            b_p = b(sum(dimarray_eq(index_arr(1:p))) - dimarray_eq(index)+1:sum(dimarray_eq(index_arr(1:p))),:);
        end

        if 0 == dimarray_ineq(index) 
            D_p = zeros(0,n_x);
            f_p = zeros(0,1);
        else
            D_p = D(sum(dimarray_ineq(index_arr(1:p))) - dimarray_ineq(index)+1:sum(dimarray_ineq(index_arr(1:p))),:);
            f_p = f(sum(dimarray_ineq(index_arr(1:p))) - dimarray_ineq(index)+1:sum(dimarray_ineq(index_arr(1:p))),:);
        end

        n_v = dimarray_ineq(index);

        H_p = [Z.' * (A_p.' * A_p) * Z,  zeros(n_x,n_v)
                zeros(n_v,n_x),       eye(n_v)];

        c_p = [Z.' * A_p.' * (A_p * x_star - b_p)
              zeros(n_v, 1);]; 
        
        D_cat = cat(1,D_cat,D_p);
        f_cat = cat(1,f_cat,f_p);
        
        n_D_cat = size(D_cat, 1);
        % 
        D_tilde = [ [zeros(n_v,n_x); D_cat * Z], ...
                    [-eye(n_v);zeros(n_D_cat-n_v,n_v);-eye(n_v)] ];

        f_tilde = [zeros(n_v, 1);
            f_cat - D_cat * x_star + [v_cat; zeros(n_v,1)]];

        H_p = H_p + regularization * eye(size(H_p));
        % ======================= Solve The QP ======================= %

        options =  optimset('Display','off');
        % options = optimoptions('quadprog','Algorithm','active-set');
        if p == 1
            x_km1 = [x_init;zeros(n_v,1)];
        else
            x_km1 = zeros(n_x+n_v,1);
        end

        [sol,~,exitflag,~] = quadprog(H_p, c_p, D_tilde, f_tilde, [], [], [], [], x_km1, options);
 

        %  check if the solver found the optimal solution
        if exitflag ~= 1
            warning("Solver cannot find the converged solution at " + p + " task!");
            return;
        end

        % ====================== Post-processing ===================== %

        % Extract x_star from the solution.
        z = zeros(n_x,1);
        z(:,1) = sol(1:n_x);

        % Update the solution of all the tasks up to now.
        x_star = x_star + Z * z;

        % Store the history of w_star for the next priority.
        % v_star_bar{priority} = sol(nx+1:end);
        v_cat = cat(1,v_cat,sol(n_x+1:end));
        % Compute the new null space projector (skipped at the last iteration).
        if (0 ~= dimarray_eq(index)) && (p ~= n_tasks)
            Z = Z * null_space_projector(A_p * Z);
            % check if the null space is empty
            if rank(Z,1e-10) == 0
                error("Null space of " + p + " task is empty!");
                return;
            end
        end
    end
end