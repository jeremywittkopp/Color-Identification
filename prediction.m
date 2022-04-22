function labels = prediction(theta, x)
    % This function will make predictions using the neural network in
    % conjunction with the forward propagation algorithm. It will then
    % compare these predictions to the provided output integer labels (if
    % given).
    %
    % INPUTS:
    %       theta (L-1x1 cell)          Weights of the NN connections
    %       X (MxN array)               Input data
    % 
    % OUTPUTS:
    %       labels (Mx1 array)          Predicted output integer labels
    
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
end