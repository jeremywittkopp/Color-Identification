function [labels, acc] = prediction(theta, x, y)
    % This function will make predictions using the neural network in
    % conjunction with the forward propagation algorithm. It will then
    % compare these predictions to the provided output integer labels (if
    % given).
    %
    % INPUTS:
    %       theta (L-1x1 cell)          Weights of the NN connections
    %       X (MxN array)               Input data
    %       y (Mx1 array)               Output integer labels (optional)
    % 
    % OUTPUTS:
    %       labels (Mx1 array)          Predicted output integer labels
    %       acc (1x1 float)             Classification accuracy (optional)
    
    % Perform forward propagation to ascertain the probabilities with which
    % the neural network thinks each data provided belongs to each class.
    % This corresponds to the output activations.
    a = forward_propagation(theta, x);
    probs = a{end};
    
    % Initialize an array of predicted output labels. Then iterate over
    % each data provided and find its maximum probability. The class that
    % has the largest probability corresponds to that which the neural
    % network most likely believes the data to belong to.
    labels = zeros(size(probs));
    for i = 1: size(x, 1)
        max_prob = max(probs(:, i));
        labels(probs(:, i) == max_prob, i) = 1;
    end
    
    % Convert the predicted ones-hot labels to integer labels.
    labels = ones_hot('integer', labels);
    
    % If the user only passed two arguments to this function, then they
    % most likely just wanted the output labels, so just set acc to zero.
    % If they passed three arguments (the last being the known output
    % labels) then provide the classification accuracy which is simply the
    % average number of times the neural network successfully predicted the
    % correct label.
    if (nargin == 2)
        acc = 0;
    elseif (nargin == 3)
        acc = mean(labels == y);
    end
end