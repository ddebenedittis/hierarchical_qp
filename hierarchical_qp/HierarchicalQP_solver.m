% function version of HierarchicalQP.m, the way of use are the same.
function  x_star_bar = HierarchicalQP_solver(A, b, C, d, we, wi, priorities, x_init)
    
    regularization = 1e-6;
    % ======================== Initialization ======================== %
    n_tasks = length(A);

    % Dimension of the optimization vector.
    nx = size(A{1}, 2);

    % Optimization vector.
    x_star_bar = zeros(nx, 1);

    % History of the slack variables, stored as a list of np.arrays.
    w_star_arr = zeros(0,1);

    % Initialize the null space projector.
    Z = eye(nx);


    C_cat = zeros(0,nx);
    d_cat =  zeros(0,1);
    
     % ======================== Check Dimensions ======================== %
    % Convert empty matrices or None into empty matrices of opportune size.
    for i = 1:n_tasks
        if isempty(A{i})
            A{i} = zeros(0,nx);
        end

        if isempty(b{i})
            b{i} = zeros(0,1);
        end

        if isempty(C{i})
            C{i} = zeros(0,nx);
        end

        if isempty(d{i})
            d{i} = zeros(0,1);
        end
    end

    % Chech that the priorities list is correctly constructed
    if ~isempty(priorities)
        for i = 1:n_tasks
            if ~ismember(i, priorities) % ?
                error("priorities is ill formed: priorities = " + ...
                    num2str(priorities))
            end
        end
    end

    % Check that the initial value dimension match with the variable
    if ~isempty(x_init)
        if length(x_init) ~= nx
            error("initial value dimension doesn't match ")
        end
    end

    if length(b) ~= n_tasks || length(C) ~= n_tasks || length(d) ~= n_tasks || ...
            (~isempty(priorities) && length(priorities) ~= n_tasks)
        error("A, b, C, d, priorities must be lists of the same length." + ...
            "Received lists of " + length(A) + ", " + ...
            length(b) + ", " + length(C) + ", " + length(d) + ...
            ", " + length(priorities) + " elements.")
    end

    for i = 1:n_tasks
        if size(A{i}, 1) ~= size(b{i}, 1)
            error("At priority " + i + ", A and b have a different number of rows.")
        end

        if size(C{i}, 1) ~= size(d{i}, 1)
            error("At priority " + i + ", C and d have a different number of rows.")
        end
    end

    for i = 1:n_tasks
        if size(A{i}, 2) ~= nx
            error("At priority " + i + ", A has " + size(A{i}, 2) + " columns instead of " + nx)
        end

        if size(C{i}, 2) ~= nx
            error("At priority " + i + ", C has " + size(C{i}, 2) + " columns instead of " + nx)
        end
    end

    if ~isempty(we)
        for p = 1:length(we)
            if ~isempty(we{p}) && length(we{p}) ~= size(A{p}, 1)
                error("At priority " + p + ", we has " + length(we{p}) + ...
                    " elements instead of " + size(A{p}, 1))
            end
        end
    end

    if ~isempty(wi)
        for p = 1:length(wi)
            if ~isempty(wi{p}) && length(wi{p}) ~= size(C{p}, 1)
                error("At priority " + p + ", wi has " + length(wi{p}) + ...
                    " elements instead of " + size(C{p}, 1));
            end
        end
    end

    % ================================================================ %
    for priority = 1:n_tasks
        % Priority of task i.
        if isempty(priorities)
            index = priority;
        else
            index = find(priorities == priority);
        end

        Ap = A{index};
        bp = b{index};

        % Scale the matrices Ap and bp by we, if we is nonempty.
        if ~isempty(we)
            if ~isempty(we{index})
                Ap = we{index} .* Ap;
                bp = we{index} .* bp;
            end
        end

        % Scale the matrices Cp and dp by wi, if wi is nonempty.
        if ~isempty(wi)
            if ~isempty(wi{index})
                C{index} = wi{index} .* C{index};
                d{index} = wi{index} .* d{index};
            end
        end

        % Slack variable dimension at task p.
        nw = size(C{index}, 1);


        % See Kinematic Control of Redundant Manipulators: Generalizing
        % the Task-Priority Framework to Inequality Task for the math
        % behind it.

        % ====================== Compute H And P ===================== %

        if ~isempty(Ap)
            H = [
                Z.' * (Ap.' * Ap) * Z,  zeros(nx,nw)
                zeros(nw,nx),       eye(nw)
                ];

            p = [
                Z.' * Ap.' * (Ap * x_star_bar - bp)
                zeros(nw, 1);
                ];
        else
            H = [
                zeros(nx,nx),  zeros(nx,nw)
                zeros(nw,nx),       eye(nw)
                ];

            p = zeros(nx+nw, 1);
        end

        % Make H positive definite
        H = H + regularization * eye(size(H));

        % ================ Compute C_tilde And D_tilde =============== %

       C_cat = cat(1,C_cat,C{index});
       d_cat = cat(1,d_cat,d{index});
 
       nC2 = size(C_cat, 1);

       C_tilde = [ zeros(nw,nx),       - eye(nw)
            C_cat * Z,   [zeros(nC2-nw,nw); - eye(nw)]
            ];

        d_tilde = [
            zeros(nw, 1);
            d_cat ...
            - C_cat * x_star_bar ...
            + [w_star_arr; zeros(nw, 1)]
            ];

        % ======================= Solve The QP ======================= %

        options =  optimset('Display','off');
        % options = optimoptions('quadprog','Algorithm','active-set');
        if priority == 1
            x_km1 = x_init;
        else
            x_km1 = zeros(nx+nw,1);
        end

        if isempty(C_tilde)
            [sol,~,exitflag,~]  = quadprog(H, p, [], [], [], [], [], [], x_km1, options);
        else
            [sol,~,exitflag,~] = quadprog(H, p, C_tilde, d_tilde, [], [], [], [], x_km1, options);
        end

        % check if the solver found the optimal solution
        if exitflag ~= 1
            warning("Solver cannot find the converged solution at " + priority + " task!");
            return;
        end

        % ====================== Post-processing ===================== %

        % Extract x_star from the solution.
        x_star = zeros(nx,1);
        x_star(:,1) = sol(1:nx);

        % Update the solution of all the tasks up to now.
        x_star_bar = x_star_bar + Z * x_star;

        % Store the history of w_star for the next priority.
        w_star_arr = cat(1,w_star_arr,sol(nx+1:end));
        
        % Compute the new null space projector (skipped at the last iteration).
        if (~isempty(Ap)) && (priority ~= n_tasks)
            Z = Z * null_space_projector(Ap * Z);
            % check if the null space is empty
            if rank(Z,1e-10) == 0
                error("Null space of " + priority + " task is empty!");
                return;
            end
        end
    end
end