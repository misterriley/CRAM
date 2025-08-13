% SUMMARY: This function computes the matrix square root of the given 
% matrix using the sqrtm function and ensures that the result has real 
% values.

function s = rsqrtm(mat)
    % Ensure symmetry
    P = (mat + mat') / 2;
    % Eigendecomposition
    [V, D] = eig(P);
    % Set any small negative eigenvalues to zero for stability
    D(D < 0) = 0;
    % Square root of the positive semi-definite part
    s = V * sqrt(D) * V';
end
