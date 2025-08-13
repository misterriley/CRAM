% SUMMARY: This code defines an enumeration class 'metric'

classdef metric
    enumeration
        % Define enumeration options
        Angular,
        Chebychev,
        Euclidean,
        Manhattan,
        Wasserstein,
        Wasserstein_n, % Wasserstein between low-rank matrix approximations

        % Not a metric, but a method
        CPM
    end
end
