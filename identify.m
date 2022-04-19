function [pred, F] = identify(color_rgb, x, theta)
    % This function will use the provided neural network to try to guess
    % the colors in the provided image.
    %
    % INPUTS:
    %       color_rbg (Cx3 array)       RGB values of each color
    %       x (Px3)                     Flattened image we want to know
    %                                   the colors of.
    %       theta (L-1x1 cell)          Weights of the NN connections
    %
    % OUTPUT:
    %       pred (Px3)                  Predicted colors of each image
    %       F (Px1)                     Predicted feature map(s)
    
    % Since the training examples were originally scaled, we should scale
    % the images as well.
    x = feature_scale(x);
    
    % Initialize space for the predictions and for the feature map(s).
    pred = zeros(size(x));
    F = zeros(size(x, 1), 1);
    
    % For each pixel in the image, determine which color it belongs to and
    % build the replication of the image with that corresponding color.
    for p = 1: size(x)
        pixel_color = prediction(theta, x(p, :));
        pred(p, :) = color_rgb(pixel_color, :);
        F(p) = pixel_color;
    end
end