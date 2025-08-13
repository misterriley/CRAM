When using this code please cite the author.

This codebase performs Connectome Regression in Alternate
Metrics (CRAM) analysis. It takes input parameters, creates necessary 
folders, calculates a distance matrix (if it does not exist or is 
malformed), performs dimensionality reduction, and recovers
significant network connections. The output directory will contain a 
file called 'report.txt' containing all the parameters, and 
cram_data.mat will contain the full dataset and outputs.

Example usage:

f = "output_directory";

cd = cram_data.load(f);

cd.cv_folds = 10;

cd.cv_repeats = 100;

cd.dataset = "example_dataset";

cd.extract_networks = true;

cd.folder = f;

cd.mats = x; % (n x n x m tensor)

cd.mtrc = metric.Wasserstein;

cd.outcomes = y; % (m-length vector)

cd.p_thresh = .05; % only necessary for CPM

cd.perm_repeats = 1000;

cd.target = "example_target"; % (aka, variable name)

cd.use_fdr = true;

cd.wass_dim = 0; % for reduced-rank FC matrix approximations while using Wasserstein 

CRAM(cd);
