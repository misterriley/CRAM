function [y_predict, consensus_mask_pos, consensus_mask_neg]=cpm_cv(x,y,pthresh,kfolds,verbose, use_pos_mask, use_neg_mask)
% Runs cross validation for CPM
% x            Predictor variable
% y            Outcome variable
% pthresh      p-value threshold for feature selection
% kfolds       Number of partitions for dividing the sample
% y_test       y data used for testing
% y_predict    Predictions of y data used for testing

% Split data
nsubs=size(x,2);
randinds=randperm(nsubs);
ksample=floor(nsubs/kfolds);

y_predict = zeros(nsubs, 1);
% Run CPM over all folds
if verbose
    fprintf('\n# Running over %1.0f Folds.\nPerforming fold no. ',kfolds);
end

consensus_mask_pos = ones(length(x),1);
consensus_mask_neg = ones(length(x),1);
n_found = 0;
for leftout = 1:kfolds
    if verbose
        fprintf('%1.0f ',leftout);
    end
    
    si = (leftout - 1) * ksample + 1;
    fi = leftout * ksample;
    if fi > nsubs || leftout == kfolds
        fi = nsubs;
    end

    testinds=randinds(si:fi);
    traininds=setdiff(randinds,testinds);
    
    % Assign x and y data to train and test groups 
    x_train = x(:,traininds);
    y_train = y(traininds);
    x_test = x(:,testinds);

    % Train Connectome-based Predictive Model
    [~, ~, pmask, mdl] = cpm_train(x_train, y_train,pthresh,use_pos_mask, use_neg_mask);
    consensus_mask_pos = consensus_mask_pos & pmask == 1;
    consensus_mask_neg = consensus_mask_neg & pmask == -1;
    
    % Test Connectome-based Predictive Model
    vals = cpm_test(x_test, mdl, pmask);
    n_found = n_found + length(vals);
    [y_predict(testinds)] = vals;
end