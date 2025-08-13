function [y_predict, performance, cm_pos, cm_neg] = cpm_main(x,y,varargin)
% Performs Connectome-Based Predictive Modeling (CPM)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   REQUIRED INPUTS
%        x            Predictor variable (e.g., connectivity matrix)
%                    Allowed dimensions are 2D (n x nsubs) OR 3D (nx m x nsubs)
%        y            Outcome variable (e.g., behavioral scores)
%                    Allowed dimensions are 2D (i x nsubs)
%        'pthresh'    p-value threshold for feature selection
%        'kfolds'     Number of partitions for dividing the sample
%                    (e.g., 2 =split half, 10 = ten fold)
%
%   OUTPUTS
%        y_predict    Predictions of outcome variable
%        performance  Correlation between predicted and actual values of y
%
%   Example:
%        [yhat,perf]=cpm_main(data,gF,'pthresh',0.05,'kfolds',2);
%
%   References:
%        If you use this script, please cite:
%        Shen, X., Finn, E. S., Scheinost, D., Rosenberg, M. D., Chun, M. M.,
%          Papademetris, X., & Constable, R. T. (2017). Using connectome-based
%          predictive modeling to predict individual behavior from brain connectivity.
%          Nature Protocols, 12(3), 506.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Parse input
p=inputParser;
defaultpthresh=0.05;
defaultkfolds=2;
defaultverbose=true;

addRequired(p,'x',@isnumeric);
addRequired(p,'y',@isnumeric); % must be n x nsubs
addParameter(p,'pthresh',defaultpthresh,@isnumeric);
addParameter(p,'kfolds',defaultkfolds,@isnumeric);
addParameter(p, 'verbose',defaultverbose,@islogical);
addParameter(p, 'use_pos_mask', true, @islogical);
addParameter(p, 'use_neg_mask', true, @islogical);

parse(p,x,y,varargin{:});

pthresh = p.Results.pthresh;
kfolds = p.Results.kfolds;
verbose = p.Results.verbose;
use_pos_mask = p.Results.use_pos_mask;
use_neg_mask = p.Results.use_neg_mask;

clearvars p

%% Check for errors
[x,y]=cpm_check_errors(x,y,kfolds);

%% Train & test Connectome-Based Predictive Model
[y_predict,cm_pos,cm_neg]=cpm_cv(x,y,pthresh,kfolds,verbose,use_pos_mask,use_neg_mask);
%% Assess performance
[performance(1),performance(2)]=corr(y_predict(:),y(:));

if verbose
    fprintf('\nDone.\n')
end
end
