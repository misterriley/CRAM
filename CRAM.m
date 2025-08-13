% SUMMARY: This function performs Connectome Regression in Alternate
% Metrics (CRAM) analysis. It takes input parameters, creates necessary 
% folders, calculates a distance matrix (if it does not exist or is 
% malformed), performs dimensionality reduction, and recovers
% significant network connections.
%
% Example usage:
% 
% f = "output_directory";
% cd = cram_data.load(f);
% cd.cv_folds = 10;
% cd.cv_repeats = 100;
% cd.dataset = "example_dataset";
% cd.extract_networks = false;
% cd.folder = f;
% cd.mats = x; % (n x n x m tensor)
% cd.mtrc = metric.CPM;
% cd.outcomes = y; % (m-length vector)
% cd.p_thresh = .05; % only necessary for CPM
% cd.perm_repeats = 1000;
% cd.target = "example_target"; % (aka, variable name)
% cd.use_fdr = true;
% cd.wass_dim = 0; % for reduced-rank FC matrix approximations while using Wasserstein 
%
% CRAM(cd);

function CRAM(cd)    
    for k = 1:80
        fprintf("V");
    end
    fprintf("\n\n");

    fprintf("Starting CRAM session on the %s dataset with %s metric\n", cd.dataset, cd.mtrc);

    start_time = tic();
    
    % Create a directory for results
    warning('off', 'all');
    status = mkdir(cd.folder);
    if ~status
        disp("folder cannot be created. exiting")
        return;
    end

    fprintf("Outputs will be stored at %s\n", cd.folder);

    if cd.mtrc ~= metric.CPM
        ensure_distmatrix(cd);
        cd.best_dim = calculate_best_dim(cd.distmatrix);
    end
    ensure_best_r(cd);

    if cd.mtrc ~= metric.CPM && cd.extract_networks
        % Recover significant network connections
        permutation_tests(cd);
    
        cd.p_regression = sum(cd.best_r < cd.perm_rs)/length(cd.perm_rs);
        cd.sig_pos_edges = sum(cd.pos_net, 'all')/2;
        cd.sig_neg_edges = sum(cd.neg_net, 'all')/2;
    end

    cd.total_time = toc(start_time);

    write_report(cd);
    cd.save();

    fprintf("\n");
    for k = 1:80
        fprintf("^")
    end
    fprintf("\n\n");
end

function write_report(cd)

    fprintf("Text of report:\n\n\n");

    report_file = fullfile(cd.folder, "report.txt");
    fID = fopen(report_file, "w");

    w(fID, "CRAM Session Report")
    w(fID, "    ----    ");

    currentDateTime = datetime('now');
    w2(fID, "Destination folder", cd.folder);
    w2(fID, "Elapsed time", sprintf("%.1f seconds", cd.total_time));
    w2(fID, "Completion time", datestr(currentDateTime, 'yyyy/mm/dd HH:MM'));
    w2(fID, "Dataset", cd.dataset);
    m_string = sprintf("%s", cd.mtrc);
    if cd.mtrc == metric.Wasserstein_n
        m_string = sprintf("%s (%d)", m_string, cd.wass_dim);
    end
    w2(fID, "Metric", m_string);
    w2(fID, "Dimension of matrix array", num2str(size(cd.mats)));
    w2(fID, "Best dimension", num2str(cd.best_dim));
    w2(fID, "R at best dimension", num2str(cd.best_r));
    w2(fID, "Rho at best dimension", num2str(cd.best_rho));
    w2(fID, "Coefficient of Determination at best dimension", num2str(cd.best_cod));
    w2(fID, "RMSE at best dimension", num2str(cd.best_rmse));
    w2(fID, "P-value for regression at best dimension", num2str(cd.p_regression));
    w2(fID, "Cross-validation folds", num2str(cd.cv_folds));
    w2(fID, "Cross-validation repeats", num2str(cd.cv_repeats));
    w2(fID, "Permutation count", num2str(cd.perm_repeats));
    w2(fID, "FDR used?", num2str(cd.use_fdr));
    w2(fID, "Threshold", num2str(cd.p_thresh));
    w2(fID, "Positive edges found", num2str(cd.sig_pos_edges));
    w2(fID, "Negative edges found", num2str(cd.sig_neg_edges));

    fclose(fID);

    fprintf("\n\nCRAM Session finished. Report written to %s.\n\n", report_file);
end

function w2(fileID, key, value)
    w(fileID, sprintf("%s: %s", key, value));
end

function w(fileID, line)
    fprintf(fileID, "%s\n", line);
    fprintf("%s\n", line);
end

function ensure_best_r(cd)
    fprintf("Calculating cross-validated r.\n");

    if isempty(gcp('nocreate'))
        parpool;
    end

    cv_outputs = zeros(cd.cv_repeats, 4);
        
    n = size(cd.mats, 1);
    cpm_mask_len = (n^2 - n)/2;
    cpm_pos_masks = zeros(cpm_mask_len, cd.cv_repeats);
    cpm_neg_masks = zeros(cpm_mask_len, cd.cv_repeats);

    % Calculate and average correlation for each cross-validation iteration
    tic
    for t = 1:cd.cv_repeats
        c = cvpartition(length(cd.outcomes), 'KFold', cd.cv_folds);
        [preds, cpm_mask_pos, cpm_mask_neg] = get_all_preds(c, cd, cd.outcomes);
        if cd.mtrc == metric.CPM
            cpm_pos_masks(:, t) = cpm_mask_pos;
            cpm_neg_masks(:, t) = cpm_mask_neg;
        end

        [RHO, ~] = corr(preds, cd.outcomes, 'Type', 'Spearman');
        [r, ~] = corr(preds, cd.outcomes, 'Type', 'Pearson');
        
        cod = r^2;
        rmse = sqrt(mean((preds - cd.outcomes).^2));
        cv_outputs(t,:) = [RHO, r, cod, rmse];
    end

    if cd.mtrc == metric.CPM
        pos_mask_flat_consensus = all(cpm_pos_masks,2);
        neg_mask_flat_consensus = all(cpm_neg_masks,2);

        pos_mask_2d_consensus = zeros(n);
        neg_mask_2d_consensus = zeros(n);

        idx = find(triu(true(n), 1));

        pos_mask_2d_consensus(idx) = pos_mask_flat_consensus;
        neg_mask_2d_consensus(idx) = neg_mask_flat_consensus;

        pos_mask_2d_consensus = pos_mask_2d_consensus + pos_mask_2d_consensus';
        neg_mask_2d_consensus = neg_mask_2d_consensus + neg_mask_2d_consensus';

        cd.pos_net = pos_mask_2d_consensus;
        cd.neg_net = neg_mask_2d_consensus;
    end
    
    cd.best_rho = mean(cv_outputs(:,1));
    cd.best_r = mean(cv_outputs(:,2));
    cd.best_cod = mean(cv_outputs(:,3));
    cd.best_rmse = mean(cv_outputs(:,4));
    
    fprintf("Cross-validated rho  = %.04f\n", cd.best_rho);
    fprintf("Cross-validated r    = %.04f\n", cd.best_r);
    fprintf("Cross-validated cod  = %.04f\n", cd.best_cod);
    fprintf("Cross-validated rmse = %.04f\n", cd.best_rmse);
    
    toc

    cd.save();
end

function best_dim = calculate_best_dim(distmatrix)
    fprintf("Calculating embedding ... \n");
    [~, best_dim] = get_embedding(distmatrix);
    fprintf("\bdone.\nOptimal dimension is %d.\n", best_dim);
end