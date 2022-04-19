function [x, y, D] = color_generation(color_names, color_rgb, M, sigma)
    % This function will generate random training data to be used to train
    % a neural network to identify colors. This neural network will then be
    % used to identify colors in an image.
    %
    % INPUTS:
    %       color_names (1xC cell)      Names of each color
    %       color_rbg (Cx3 array)       RGB values of each color
    %       M (1x1 int)                 Number of examples to
    %                                   generate for each color
    %       sigma (1x1 float)           Spread in the examples from the
    %                                   actual RGB value
    %
    % OUTPUTS:
    %       x (MxC array)               Input training examples
    %       y (Mx1 array)               Output integer labels
    %       D (CxC array)               Similarity of the colors
    
    % Get the number of colors that the user wants to identify and get the
    % number of channels each image will have (this last option is just so
    % that if the user wants to use a different representation such as CMKY
    % they can).
    C = length(color_names);
    N = size(color_rgb, 2);
    
    % Initialize the input matrix and output matrix. Note that for each
    % color, their will be M examples. Thus there will be a total of M*C
    % training examples for the neural network.
    x = zeros(M*C, N);
    y = zeros(M*C, 1);
    
    % Iterate over each color and generate M random RGB values. If the
    % current color's RGB value includes a zero, then the random
    % distribution will be a uniform one. Otherwise, the random RGB value
    % will be drawn from a normal distribution centered around the original
    % value with a spread of sigma. Note that since RGB values are defined
    % on the interval [0, 255], then the output will be forced onto this
    % interval. Also, assign each example to the corresponding class.
    for c = 1: C
        for i = 1: M
            ind = sub2ind([M, C], i, c);
            for j = 1: N
                x(ind, j) = rand_val(color_rgb(c, j), sigma);
                y(ind) = c;
            end
        end
    end
    
    % Randomly shuffle the data so the neural network doesn't learn the
    % order of the data instead of the data itself.
    idx = randperm(size(x, 1));
    x = x(idx, :);
    y = y(idx);
    
    % Initialize a matrix that will contain the normalized distances of
    % each color from one another. These distances will be measured using
    % the taxicab metric since the RGB values live in Z+3. The normalized
    % distances will provide a measure of how similar the colors are,
    % providing a glimpse as to how well the neural network can learn these
    % colors.
    D = zeros(C, C);
    for i = 1: C
        for j = 1: C
            D(i, j) = sum(abs(color_rgb(i,:) - color_rgb(j,:)))./(3*255);
        end
    end
end



function val = rand_val(mu, sigma)
    % This supplimentary function will randomly generate an RGB value that
    % is centered around mu with spread sigma that keeps the values on the
    % interval [0, 255].
    % 
    % INPUTS:
    %       mu (1x1 int)            Original RBG value
    %       sigma (1x1 float)       Spread that the generated values can
    %                               have relative to the original value
    % 
    % OUTPUT:
    %       val (1x1 int)           Generated RGB value
    
    % Handle the case where the channel's original value is greater than
    % zero (i.e., that channel is contributing something to the color).
    if (mu > 0)
        
        % Sample a normal distribution with mean mu and standard deviation
        % sigma. Then, since RGB values are positive integers, round the
        % randomly sampled value to the nearest ones' digit.
        val = round(mu + sigma * randn());
        
        % Handle the case where the generated value exceeds the upper limit
        % of the interval [0, 255]. In this case, figure out how much this
        % value exceeds the upper limit by and subtract the generated value
        % by two times that excess.
        if (val > 255)
            d = val - 255;
            val = val - 2*d;
            
        % Handle the case where the generated value is below the lower
        % limit of the interval [0, 255]. In this case, simplfy take the
        % absolute value of the negative value.
        elseif (val < 0)
            val = abs(val);
        end
        
    % Handle the case where the channel's original value is zero (i.e., 
    % that channel is not contributing anything to the color). In this
    % case, then just randomly sample a uniform distribution on the
    % interval [0, sigma].
    elseif (mu == 0)
        val = round(sigma * rand());
    end
end