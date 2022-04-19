function theta = initialize_weights(eps, N)
    % This function will randomly initialize the weights of the neural 
    % network where the weights are drawn from a uniform distribution on
    % the interval [-eps, eps].
    %
    % INPUTS:
    %       eps (1x1 float)             Parameter governing initialized
    %                                   weight random variable distribution
    %       N (Lx1 array)               Topology of the neural network
    % 
    % OUTPUT:
    %       theta (L-1x1 cell)          Weights of the NN connections
    
    % Initialize the cell array containing all the weights of the neural
    % network. This will provide an efficient way of communicating this
    % information between the various functions. Then randomly initialize
    % each of the weight matrices, where each element is drawn from a
    % uniform distribution on the interval [-eps, eps].
    theta = cell(length(N)-1, 1);
    for l = 1: length(theta)
        theta{l} = 2 .* eps .* rand(N(l+1), N(l) + 1) - eps;
    end
end