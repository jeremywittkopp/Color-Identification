function x_norm = feature_scale(x)
    % This function will perform feature scaling on a data set. This will
    % first do mean normalization, then it will rescale the data using the
    % standard deviation.
    %
    % INPUT:
    %       x (MxN array)       Input data unscaled
    %
    % OUTPUT:
    %       x_norm (MxN array)  Input data that has been mean normalized
    %                           and rescaled
    
    % Compute the mean and standard deviation of each of the features in
    % the input data.
    mu = mean(x, 1);
    sigma = std(x, 1);
    
    % Perform mean normalization and rescale the input data.
    x_norm = (x - mu) ./ sigma;
end