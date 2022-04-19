function theta = backward_propagation(alpha, theta, x, y)
    % This function will implement the backpropagation algorithm used to
    % update the weights of the neural network during the training process.
    %
    % INPUTS:
    %       alpha (1x1 float)           Learning rate
    %       theta (L-1x1 cell)          Weights of the NN connections
    %       x (MxN array)               Input training examples
    %       y (Mx1 array)               Output integer labels
    %
    % OUTPUT:
    %       theta (L-1x1 cell)          Weights of the NN connections
    
    % Get the number of layers in the neural network and the number of
    % training examples that will be used to train the network with.
    L = length(theta) + 1;
    M = size(x, 1);
    
    % Convert the integer output labels to ones-hot notation. This function
    % will ensure that the dimensions line up when performing the
    % backpropagation algorithm.
    y = ones_hot('ones-hot', y);
    
    % Perform the forward propagation algorithm in order to see what
    % class(es) the training examples belong to with its current
    % weightings.
    a = forward_propagation(theta, x);
    
    % Initialize a cell array containing the error terms for each layer
    % (minus the input layer, since that is assumed to be correct).
    delta = cell(L-1, 1);
    
    % Iterate backwards from the output layer to the first hidden layer
    % (the layer after the input layer). If we ae considering the output
    % layer, then the error is simply the absolute error between the
    % original output matrix and its activations. If we are considering one
    % of the hidden layers, then (using a lot of the chain rule) we want to
    % find how the weights caused this error to arise. Note that each error
    % term needs to be stripped of the constant row.
    for l = L: -1: 2
        if (l == L)
            delta{l-1} = a{l} - y;
        else
            delta{l-1} = (theta{l}' * delta{l}) .* a{l} .* (1 - a{l});
            delta{l-1} = delta{l-1}(2: end, :);
        end
    end

    % Iterate over each of the weights and update them using the gradient
    % computed earlier. This corresponds to updating the weights so they
    % (hopefully) won't make the same error as they did when we made the
    % predictions using forward propagation.
    for l = 1: length(theta)
        theta{l} = theta{l} - alpha .* (delta{l} * a{l}') ./ M;
    end
end