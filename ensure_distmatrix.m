function ensure_distmatrix(cd)
    if ~cram_data.check_for_variable(cd, "distmatrix", true)
        calc_distmatrix(cd);
    end
end

function calc_distmatrix(cd)
    switch(cd.mtrc)
        case metric.Wasserstein
            cd.distmatrix = calc_wass_distmatrix(cd.mats, 0);
        case metric.Wasserstein_n
            cd.distmatrix = calc_wass_distmatrix(cd.mats, cd.wass_dim);
        case metric.Euclidean
            cd.distmatrix = calc_norm_distmatrix(cd.mats, 2);
        case metric.Manhattan
            cd.distmatrix = calc_norm_distmatrix(cd.mats, 1);
        case metric.Chebychev
            cd.distmatrix = calc_norm_distmatrix(cd.mats, inf);
        case metric.Angular
            cd.distmatrix = calc_angle_distmatrix(cd.mats);
        otherwise
            error("Unimplemented distance metric: %s", string(cd.mtrc))
    end

    cd.distmatrix(1:size(cd.distmatrix, 1) + 1:end) = 0;
    % in case there are complex values due to numerical instability
    cd.distmatrix = real(cd.distmatrix);

    disp("distance matrix calculated");
    
    cd.save();
end

function distmatrix = calc_angle_distmatrix(mats)
    fprintf("Calculating angular distance matrix\n");
    % Distance should be based on fisher-transformed values
    mats = ensure_fisher_transform(mats, true);

    n = size(mats, 3);
    distmatrix = zeros(n);
    parfor i = 1:n
        mat1 = mats(:,:,i);
        m1 = mat1(triu(true(size(mat1)), 1));
        for j = 1:n
            if j > i
                mat2 = mats(:,:,j);
                m2 = mat2(triu(true(size(mat2)), 1));
                
                cosTheta = dot(m1, m2) / (norm(m1) * norm(m2));
                theta = acos(cosTheta);
                distmatrix(i,j) = theta;
            end
        end
    end

    distmatrix = distmatrix + distmatrix';
end

function distmatrix = calc_norm_distmatrix(mats, p)

    fprintf("Calculating distance matrix with p-norm %d\n", p);
    % Distance should be based on fisher-transformed values
    mats = ensure_fisher_transform(mats, true);

    n = size(mats, 3);
    distmatrix = zeros(n);
    parfor i = 1:n
        mat1 = mats(:,:,i);
        for j = 1:n
            if j > i
                mat2 = mats(:,:,j);
                d = norm(mat1(:) - mat2(:), p);
                distmatrix(i,j) = d;
            end
        end
    end

    distmatrix = distmatrix + distmatrix';
end

function distmatrix = calc_wass_distmatrix(mats, k)
    % Need raw correlations, not fisher transforms for the Wasserstein
    mats = ensure_fisher_transform(mats, false);
    m = size(mats, 3);  % Number of matrices
    n = size(mats, 1);
    if k == 0 % zero is a sentinel value saying to use the whole matrix
        k = n;
    end
    
    % Initialize the distance matrix
    distmatrix = zeros(m, m);

    A_ks = cell(m,1);
    sqrts = cell(m,1);
    parfor i = 1:m
        A = mats(:,:,i);
        [V_A, D_A] = eigs(A, k, 'largestabs');
        lambda_A_k = max(diag(D_A), 0); % in case of numerical stability problems setting eigenvalues to less than zero
        A_k = V_A * diag(lambda_A_k) * V_A';
        A_ks(i) = {A_k};
        sqrt_lambda_A_k = sqrt(lambda_A_k); 
        sqrtA_k = V_A * diag(sqrt_lambda_A_k) * V_A';
        sqrts(i) = {sqrtA_k};
    end
    
    indexes1 = 1:m;
    indexes2 = 1:m;
    [X,Y] = meshgrid(indexes1, indexes2);
    coords = zeros(m * m, 2);
    coords(:,1) = reshape(X,[],1);
    coords(:,2) = reshape(Y,[],1);
    coords = coords(coords(:,1) < coords(:,2),:);
    outcomes = zeros(length(coords),3);
    fprintf("calculating distance matrix with %d unique entries\n", length(coords));

    parfor i = 1:length(coords)
        warning('off', 'all');
        x = coords(i, 1);
        y = coords(i, 2);

        A_k = A_ks{x};
        sqrtA_k = sqrts{x};
        traceA_k = trace(A_k);

        B_k = A_ks{y};
        innerMatrix_k = sqrtA_k * B_k * sqrtA_k;
        innerMatrixSqrt_k = rsqrtm(innerMatrix_k);
    
        traceB_k = trace(B_k);
        traceInner_k = trace(innerMatrixSqrt_k);
        d_Wass_low_rank = sqrt(traceA_k + traceB_k - 2 * traceInner_k);
        outcomes(i,:) = [x,y,d_Wass_low_rank];
    end

    for i = 1:length(outcomes)
        x = outcomes(i,1);
        y = outcomes(i,2);
        dist = outcomes(i,3);
        distmatrix(x, y) = dist;
        distmatrix(y, x) = dist;
    end
end
