function J = cost_function(theta, x, y)
    % This function will compute the cross-entropy cost function for
    % training the neural network.
    %
    % INPUTS:
    %       theta (L-1x1 cell)          Weights of the NN connections
    %       X (MxN array)               Input data
    %       y (Mx1 array)               Output integer labels (optional)
    %
    % OUTPUT:
    %       J (1x1 float)               Cross-entropy cost function for
    %                                   using the weights theta
    
    % Convert the integer output labels to ones-hot notation. This function
    % will ensure that the dimensions line up when performing the
    % backpropagation algorithm.
    y_oh = ones_hot('ones-hot', y);
    
    % Perform the forward propagation algorithm in order to see what
    % class(es) the training examples belong to with its current
    % weightings.
    a = forward_propagation(theta, x);
    
    % Compute the cross-entropy cost function of the predictions to see how
    % different the two distributions are.
    J = mean(sum(-y_oh .* log(a{end}) - (1-y_oh) .* log(1-a{end}), 1));
end