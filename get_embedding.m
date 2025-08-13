% SUMMARY: This function implements the landmark MDS (Multi-Dimensional 
% Scaling) algorithm to compute the embedding of data points based on a 
% distance matrix.

function [L, dim] = get_embedding(distmatrix)
    [L, e] = cmdscale(distmatrix, length(distmatrix));
    dim = round(find(e > mean(e), 1, 'last'), 0);    
end

function embedding = landmark_MDS()
    % Calculate the embedding using the landmark MDS algorithm
    
    % Extract the distance matrix for the landmarks
    landmark_distmatrix = distmatrix(1:n_landmarks, 1:n_landmarks);
    landmark_distmatrix_sq = landmark_distmatrix.^2;
    
    % Perform classical MDS on the landmark distance matrix
    [landmark_embedding, ~] = cmdscale(landmark_distmatrix, dim);
    
    % Calculate the pseudo-inverse of the landmark embedding
    l_sharp = pinv(landmark_embedding);
    
    % Calculate squared distances for non-landmark points
    landmark_to_other_distmatrix = distmatrix(1:n_landmarks, n_landmarks + 1:end);
    landmark_to_other_distmatrix_sq = landmark_to_other_distmatrix.^2;
    
    % Calculate delta_mu (mean squared distance) for landmarks
    delta_mu = mean(landmark_distmatrix_sq, 2);
    
    % Calculate embedding for non-landmark points using the landmark MDS 
    % embedding
    other_embedding = 1/2 * l_sharp * ...
        (repmat(delta_mu, 1, size(landmark_to_other_distmatrix, 2)) - ...
        landmark_to_other_distmatrix_sq);
    
    % Concatenate landmark and non-landmark embeddings
    ret = vertcat(landmark_embedding, other_embedding');
end
