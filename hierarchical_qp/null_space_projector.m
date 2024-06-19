function N = null_space_projector(A)
    nx = size(A, 2);
    N = eye(nx) - pinv(A) * A;
end