function [r,p,pmask,mdl]=cpm_train(x,y,pthresh,use_pos_mask,use_neg_mask)
% Train a Connectome-based Predictive Model
% x            Predictor variable
% y            Outcome variable
% pthresh      p-value threshold for feature selection
% r            Correlations between all x and y
% p            p-value of correlations between x and y
% pmask        Mask for significant features
% mdl          Coefficient fits for linear model relating summary features to y

% Select significant features
[r,p]=corr(x',y);
if use_pos_mask
    if use_neg_mask
        pmask=(+(r>0))-(+(r<0)); 
    else
        pmask = (+(r>0));
    end
else
    pmask = -(+(r<0));
end
pmask=pmask.*(+(p<pthresh));

% For each subject, summarize selected features 
for i=1:size(x,2)
    summary_feature(i)=nansum(x(pmask>0,i))-nansum(x(pmask<0,i));
end

try
    % Fit y to summary features
    mdl=robustfit(summary_feature(:),y(:));
catch ME
    disp(ME.message);
end

[warnMessage, ~] = lastwarn();
if ~isempty(warnMessage)
    %disp(warnMessage);
    lastwarn('','');
end