% ============================================================================ %
%               HIERARCHICAL QUADRATIC PROGRAMMING IMPLEMENTATION              %
% ============================================================================ %

% A general task T can be defined as
%       [ we * (A x - b)  = v
%   T = [
%       [ wi * (C x - d) <= w

% where v and w are slack variables.

% Is is formulated as a QP problem
%   min_x 1/2 (A x - b)^2 + 1/2 w^2
%   s.t.: C x - d <= w


% It can be rewritten in the general QP form:
%   min_x 1/2 xi^T H xi + p^T xi
%   s.t: CI xi + ci0 >= 0

% where:
%   H   =   A^T A
%   p   = - A^T b
%   CI  = [ -C, 0 ]
%         [  0, I ]
%   ci0 = [ d ]
%         [ 0 ]
%   xi  = [ x ]
%         [ w ]

% ============================================================================ %

% Given a set of tasks T1, ..., Tn, for the task Tp the QP problem becomes:
%   H   = [ Zq^T Ap^T Ap Zq, 0 ]
%         [               0, I ]
%   p   = [ Zq^T Ap^T (Ap x_opt - bp) ]
%         [                         0 ]

%   CI  = [   0,       I      ]
%         [ - C_stack, [0; I] ]
%   ci0 = [ 0                                    ]
%         [ d - C_stack x_opt + [w_opt_stack; 0] ]

% The solution of the task with priority p is x_p_star

% The solution of the tasks with priority equal or smaller that p+1 is
% x_p+1_star_bar = x_p_star_bar + Z @ x_p+1_star


% ============================================================================ %

classdef HierarchicalQP
    properties
        regularization = 1e-6;
    end
    
    methods(Static)
        function N = null_space_projector(A)
            nx = size(A, 2);
            N = eye(nx) - pinv(A) * A;
        end

        function [A, b, C, d] = check_dimensions(A, b, C, d, we, wi, priorities)
            % CHECK_DIMENSIONS
            %   Raise ValueError if the dimension of the input matrices are
            %   not consistent. Additonally, empty matrices are converted
            %   into empty matrices of opportune size.

            arguments
                A cell
                b cell
                C cell
                d cell
                we cell = {}
                wi cell = {}
                priorities = []
            end
            
            n_tasks = length(A);
            
            nx = size(A{1}, 2);
            
            % Convert empty matrices or None into empty matrices of opportune size.
            for i = 1:n_tasks
                if isempty(A{i})
                    A{i} = zeros(0,nx);
                end
                    
                if isempty(b{i})
                    b{i} = zeros(0);
                end
                
                if isempty(C{i})
                    C{i} = zeros(0,nx);
                end
                    
                if isempty(d{i})
                    d{i} = zeros(0);
                end
            end
                    
            % Chech that the priorities list is correctly constructed
            if ~isempty(priorities)
                for i = 1:n_tasks
                    if ismember(i, priorities)
                        error("priorities is ill formed: priorities = " + ...
                            num2str(priorities))
                    end
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
                    error("At priority" + i + ", A and b have a different number of rows.")
                end
                    
                if size(C{i}, 1) ~= size(d{i}, 1)
                    error("At priority" + i + ", C and d have a different number of rows.")
                end
            end

            for i = 1:n_tasks
                if size(A{i}, 2) ~= nx
                    error("At priority" + i + ", A has " + size(A{i}, 2) + " columns instead of " + nx)
                end
                    
                if size(C{i}, 2) ~= nx
                    error("At priority" + i + ", C has " + size(C{i}, 2) + " columns instead of " + nx)
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
        end
    end

    methods
        function x_star_bar = solve(obj, A, b, C, d, we, wi, priorities)
            % SOLVE Solve the hierarchical Quadratic Programming problem.
            %   x_star_bar = obj.solve(A, b, C, d, we, wi, priorities)
            %
            %   Given a set of tasks in the form
            %       Ap x  = b
            %       Cp x <= d,
            %   with p = 1:p_max, returns the optimal vector x_star that solves
            %   solves the hierarchical QP problem.
            %
            % Inputs:
            %   A : cell array of Ap matrices of size (ne_p, nx)
            %   b : cell array of bp vectors of size (ne_p)
            %   C : cell array of Cp matrices of size (ni_p, nx)
            %   d : cell array of dp vectors of size (ni_p)
            %   we: cell array of we_p vectors of size (ne_p)
            %   wi: cell array of wi_p vectors of size (ni_p)
            %   priorities: vector of ints representing the priorities of the
            %               tasks, from 1 to p_max
            %
            % Outputs:
            %   x_star_bar: optimal solution vector

            arguments
                obj
                A cell
                b cell
                C cell
                d cell
                we cell = {}
                wi cell = {}
                priorities = []
            end
            
            
            % ======================== Initialization ======================== %

            % Number of tasks.
            n_tasks = length(A);

            % Dimension of the optimization vector.
            nx = size(A{1}, 2);

            % Optimization vector.
            x_star_bar = zeros(nx, 1);
            
            % History of the slack variables, stored as a list of np.arrays.
            w_star_bar = {zeros(0, 1)};

            % Initialize the null space projector.
            Z = eye(nx);
            
            
            [A, b, C, d] = obj.check_dimensions(A, b, C, d, we, wi, priorities);


            % ================================================================ %

            for i = 1:n_tasks
                % Priority of task i.
                if isempty(priorities)
                    priority = i;
                else
                    priority = find(priorities == i);
                end
                
                Ap = A{priority};
                bp = b{priority};
                
                % Scale the matrices Ap and bp by we, if we is nonempty.
                if ~isempty(we)
                    if ~isempty(we{priority})
                        Ap = we{priority} .* Ap;
                        bp = we{priority} .* bp;
                    end
                end
                       
                % Scale the matrices Cp and dp by wi, if wi is nonempty.
                if ~isempty(wi)
                    if ~isempty(wi{priority})
                        C{priority} = wi{priority} .* C{priority};
                        d{priority} = wi{priority} .* d{priority};
                    end
                end
                
                % Slack variable dimension at task p.
                nw = size(C{priority}, 1);
                
                
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
                        zeros(nw, 1)
                    ];
                else
                    H = [
                        zeros(nx,nx),  zeros(nx,nw)
                        zeros(nw,nx),       eye(nw)
                    ];

                    p = zeros(nx+nw, 1);
                end
                    
                % Make H positive definite
                H = H + obj.regularization * eye(size(H));
                
                % ================ Compute C_tilde And D_tilde =============== %

                nC2 = size(cat(1, C{1:priority}), 1);

                C_tilde = [
                                 zeros(nw,nx),       - eye(nw)
                    cat(1, C{1:priority}) * Z,   zeros(nC2,nw)
                ];
                if nw > 0
                    C_tilde(end-nw+1:end, end-nw+1:end) = - eye(nw);
                end

                % w_star_arr = [w_star[priority], w_star[priority-1], ..., w_star[0]]
                w_star_arr = cat(1, w_star_bar{:});

                d_tilde = [
                    zeros(nw, 1)
                    cat(1, d{1:priority}) ...
                        - cat(1, C{1:priority}) * x_star_bar ...
                        + [w_star_arr; zeros(nw, 1)]
                ];


                % ======================= Solve The QP ======================= %

                options =  optimset('Display','off');
                if isempty(C_tilde)
                    sol = quadprog(H, p, [], [], [], [], [], [], [], options);
                else
                    sol = quadprog(H, p, C_tilde, d_tilde, [], [], [], [], [], options);
                end


                % ====================== Post-processing ===================== %

                % Extract x_star from the solution.
                x_star = sol(1:nx);
                
                % Update the solution of all the tasks up to now.
                x_star_bar = x_star_bar + Z * x_star;

                % Store the history of w_star for the next priority.
                if priority == 1
                    w_star_bar = {sol(nx+1:end)};
                else
                    w_star_bar{end+1} = sol(nx+1:end);
                end

                % Compute the new null space projector (skipped at the last iteration).
                if (~isempty(Ap)) && (priority ~= n_tasks)
                    Z = Z * obj.null_space_projector(Ap * Z);
                end
            end
        end
    end
end