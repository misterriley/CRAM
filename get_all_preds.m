% SUMMARY: This code contains functions to calculate predictions for 
% cross-validation.

function [predictions, cpm_mask_pos, cpm_mask_neg] = get_all_preds(partition, cd, outcomes)
    cpm_mask_pos = [];
    cpm_mask_neg = [];
    if cd.mtrc == metric.CPM
        [predictions, ~, cpm_mask_pos, cpm_mask_neg] = cpm_main(cd.mats, outcomes, 'pthresh', cd.p_thresh, 'kfolds', cd.cv_folds, 'verbose', false);
    else
        % Calculate predictions for all cross-validation folds
        predictions = zeros(length(cd.outcomes), 1);
        
        for i = 1:cd.cv_folds
            fold_preds = get_fold_preds(training(partition, i), cd, outcomes);
            predictions(test(partition, i)) = fold_preds;
        end
    end
end

function ret = get_fold_preds(i_train, cd, outcomes)
    % Calculate predictions for a specific fold using linear regression
    % model
    warning('off', 'stats:LinearModel:RankDefDesignMat');
    train_distmatrix = cd.distmatrix(i_train, i_train);
    [L,dim] = get_embedding(train_distmatrix);
    cd.best_dim = dim;
    train_embedding = L(:,1:dim);
    test_embedding = get_remaining_embedding_from_landmarks(train_embedding, i_train, cd.distmatrix);
    lm = fitlm(train_embedding, outcomes(i_train));
  
    ret = lm.predict(test_embedding);
end

function embedding = get_remaining_embedding_from_landmarks(landmark_embedding, ...
    landmark_locs, full_distmatrix)

    landmark_distmatrix_sq = full_distmatrix(landmark_locs, landmark_locs).^2;

    % Calculate the pseudo-inverse of the landmark embedding
    l_sharp = pinv(landmark_embedding);
    
    % Calculate squared distances for non-landmark points
    landmark_to_other_distmatrix = full_distmatrix(landmark_locs, ~landmark_locs);
    landmark_to_other_distmatrix_sq = landmark_to_other_distmatrix.^2;
    
    % Calculate delta_mu (mean squared distance) for landmarks
    delta_mu = mean(landmark_distmatrix_sq, 2);
    
    % Calculate embedding for non-landmark points using the landmark MDS 
    % embedding
    embedding = 1/2 * l_sharp * ...
        (repmat(delta_mu, 1, size(landmark_to_other_distmatrix, 2)) - ...
        landmark_to_other_distmatrix_sq);
    embedding = embedding';
end
