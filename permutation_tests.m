% SUMMARY: This code recovers a network by computing the correlation 
% between features and model predictions, and performing permutation 
% testing to check for significance of edges in the network.

function permutation_tests(cd)

    if cd.perm_repeats == 0
        return;
    end

    do_tests = ~cram_data.check_for_variable(cd, "pos_net", true) && ...
        ~cram_data.check_for_variable(cd, "neg_net", true) && ...
        ~cram_data.check_for_variable(cd, "permutation_rs", true);

    if ~do_tests
        %return
    end

    % Data should be fisher transformed
    cd.mats = ensure_fisher_transform(cd.mats, true);

    % Initialize slopes array for storing slope values
    permuted_edge_rs = zeros(size(cd.mats, 1), size(cd.mats, 2), cd.perm_repeats);

    % Partition data for cross-validation and calculate true slopes
    partition = cvpartition(length(cd.outcomes), 'KFold', cd.cv_folds);
    true_predictions = get_all_preds(partition, cd, cd.outcomes);
    true_edge_rs = get_all_rs(cd.mats, true_predictions);

    if isempty(gcp('nocreate'))
        parpool;
    end

    % Display information about permutation testing
    disp("Running permutation tests to recover network");

    permutation_rs = zeros(cd.perm_repeats, 1);
    tic

    parfor repeat_index = 1:cd.perm_repeats
        % Permute outcomes, partition data, and calculate slopes for permutations
        permuted_outcomes = cd.outcomes(randperm(length(cd.outcomes)));
        partition = cvpartition(length(cd.outcomes), 'KFold', cd.cv_folds);
        permuted_predictions = get_all_preds(partition, cd, permuted_outcomes);
    
        if cd.extract_networks
            permuted_edge_rs(:, :, repeat_index) = get_all_rs(cd.mats, permuted_predictions);
        end
        [permutation_rs(repeat_index), ~] = corr(permuted_predictions, permuted_outcomes, 'Type', 'Spearman');
    end
    toc
    cd.perm_rs = permutation_rs;
    
    % Initialize array for p-values and calculate p-values for each edge
    p_vals = nan(size(permuted_edge_rs, 1), size(permuted_edge_rs, 2));
    for i = 1:size(p_vals, 1)
        for j = i + 1:size(p_vals, 2)
            sample = squeeze(permuted_edge_rs(i, j, :));
            ts = true_edge_rs(i,j);

            if std(sample) == 0
                % no differences in sample, this node is bogus
                continue;
            else
                p_greater = (sum(sample > ts) + .5*sum(sample==ts))/length(sample);
                p_lesser = (sum(sample < ts) + .5*sum(sample==ts))/length(sample);
                p = min(p_greater,p_lesser)*2;
                p_vals(i, j) = p;
            end
        end
    end

    % Sort and process p-values, perform False-Discovery Rate control if requested
    p_array = p_vals(:);
    p_array = p_array(~isnan(p_array));
    p_array = sort(p_array);

    if cd.use_fdr
        % False Discovery Rate check - 
        % https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Yekutieli_procedure
        p_limit = 0;
        n_sig = 0;
        m = length(p_array);
        c_m = log(m) + .57721 + 1 / (2 * m);
        for k = 1:m
            p_k = p_array(k);
            if p_k <= k * cd.p_thresh / (m * c_m)
                p_limit = p_k;
                n_sig = k;
            end
        end
    else
        p_limit = cd.p_thresh;
        n_sig = sum(p_array <= p_limit);
    end

    % Display results and generate significant network edges
    fprintf("Permutation testing complete. %d significant edges at p = %f level.\n", n_sig, p_limit);
    significant_edges = ~isnan(p_vals) & p_vals <= p_limit;
    significant_edges = significant_edges + significant_edges';

    pos_net = significant_edges;
    pos_net(true_edge_rs <= 0) = 0;
    cd.pos_net = pos_net + pos_net';

    neg_net = significant_edges;
    neg_net(true_edge_rs >= 0) = 0;
    cd.neg_net = neg_net + neg_net';

    %cd.save();
end

function rs = get_all_rs(mats, predictions)
    % Calculate regression slopes for all edges based on the model predictions
    rs = zeros(size(mats, 1), size(mats, 2));
    for i = 1:size(mats, 1)
        for j = i + 1:size(mats, 2)
            x = squeeze(mats(i, j, :));
            areAllValuesSame = all(x == x(1));
            if areAllValuesSame
                rs(i, j) = 0;
            else
                pf = polyfit(x, predictions(:), 1);
                rs(i, j) = pf(1);
            end
        end
    end
end
