function [x_train, x_test, y_train, y_test] = split_data(x, y, s)
    M = size(x, 1);
    idx = randperm(M);
    x = x(idx, :);
    y = y(idx);
    
    Mtrain = floor((1 - s) * M);
    x_train = x(1: Mtrain, :);
    y_train = y(1: Mtrain);
    
    x_test = x(Mtrain+1: end, :);
    y_test = y(Mtrain+1: end, :);
end