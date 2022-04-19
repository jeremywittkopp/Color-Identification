clear;
clc;
close all;
format compact;

% Load the color information - the color names (the output classes) and
% their corresponding RGB values. These RGB values will be used to "redraw"
% the image using only those colors.
load('color-info.mat');

% Generate random training data based on the desired colors.
[X, y, D] = color_generation(color_names, color_rgb, 200, 2);

% Establish the topology of the neural network and scale the input
% features. This will help the neural network learn more efficiently.
N = [size(X, 2), 20, max(y)];
X = feature_scale(X);

% Specify the learning rate and the number of iterations to train the
% neural network on the data.
N_iter = 800;
alpha = 2;

% Randomly initialize the weights of the neural network. Also initialize 
% the cost function evolution as a function of number of iterations and the
% classification accuracy as a function of number of iterations.
theta = initialize_weights(0.01, N);
cost = zeros(N_iter, 1);
acc = zeros(N_iter, 1);

% Train the neural network on the data. Each time, perform forward
% propagation to make a new set of predictions, determine how different the
% two distributions are via their cross-entropy and compute the
% classification accuracy, then backpropagation the errors in order to
% update the weights.
for i = 1: N_iter
    theta = backward_propagation(alpha, theta, X, y);
    [~, acci] = prediction(theta, X, y);
    acc(i) = acci;
    cost(i) = cost_function(theta, X, y);
    
    if (mod(i, 100) == 0)
        fprintf('Iteration :: %d \t\t Cost: %.3e', i, cost(i));
        fprintf('\t\t Acc. %.3e\n', acc(i));
    end
end

figure;
subplot(1, 2, 1);
plot(1: N_iter, cost);
xlabel('Number of Iterations');
ylabel('Cross Entropy');
axis xy;
axis square;

subplot(1, 2, 2);
plot(1: N_iter, acc);
xlabel('Number of Iterations');
ylabel('Classification Accuracy');
axis xy;
axis square;

% Clear some room in memory
clear acc acci alpha cost N N_iter i X y

% Save the results of the training to be used to process images.
save('color-neural-network.mat');