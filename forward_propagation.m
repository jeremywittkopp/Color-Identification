function a = forward_propagation(theta, x)
    % This function will implement the forward propagation algorithm used
    % to make predictions with the neural network.
    %
    % INPUTS:
    %       theta (L-1x1 cell)          Weights of the NN connections
    %       X (MxN array)               Input training examples
    %
    % OUTPUT:
    %       a (Lx1 cell)                Activations of each layer for each
    %                                   training example
    
    % Get the number of layers in the neural network and the number of
    % training examples that will be used to train the network with.
    L = length(theta) + 1;
    M = size(x, 1);
    
    % I will assume that each layer has the same activation function - the
    % sigmoid activation function.
    g = @(z) 1 ./ (1 + exp(-z));
    
    % Initialize the cell array of all the layers' activations. The input
    % layer's activations will consist of the input training examples along
    % with a row of ones (for the constant term).
    a = cell(L, 1);
    a{1} = [ones(1, M); x'];
    
    % Iterate over each set of connections between each of the layers of
    % the neural network. During each step, multiply the weights of the
    % connections by the previous layer's activiations. Then feed this
    % preactivation into the activation function to get the next layer's
    % activations (also, unless we are at the output layer, add an extra
    % row of ones for the constant term to the beginning).
    for l = 1: L-1
        z = theta{l} * a{l};
        a{l+1} = g(z);
        if (l < L-1)
            a{l+1} = [ones(1, M); a{l+1}];
        end
    end
end