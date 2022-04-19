clear;
clc;
close all;
format compact;

% Load the pre-trained color neural network weights, color names and RGB
% values, and the similarity matrix. Also load the image data (they should
% have been originally 250x250x3 which when flattened becomes 62500x3).
load('color-neural-network.mat');
load('image-data.mat');

% Preallocate space for the original images and the new images with the
% predicted colors and their corresponding feature maps.
imgs = zeros(250, 250, 3, 4);
preds = zeros(size(imgs));
F = zeros(250, 250, 4);

% Iterate over each image in the dataset and predict the color of each
% pixel. Fill the new image with these predicted colors. Then finally
% reshape the two images and the feature map back to 250x250.
for i = 1: size(img, 3)
    [pred, f] = identify(color_rgb, img(:, :, i), theta);
    imgs(:, :, :, i) = reshape(img(:, :, i), [250, 250, 3]);
    preds(:, :, :, i) = reshape(pred, [250, 250, 3]);
    F(:, :, i) = reshape(f, [250, 250]);
end

% Convert these back to uint8.
imgs = uint8(imgs);
preds = uint8(preds);

figure;
for i = 1: size(img, 3)
    qi = 1 + 3 * (i - 1);
    
    subplot(2, 6, qi);
    imshow(imgs(:, :, :, i));
    title(['Original ', num2str(i)]);
    axis square;
    set(gca, 'XTick', [], 'YTick', []);

    subplot(2, 6, qi+1);
    imshow(preds(:, :, :, i));
    title(['Colors Identified ', num2str(i)]);
    axis square;
    set(gca, 'XTick', [], 'YTick', []);
    
    subplot(2, 6, qi+2);
    imagesc(F(:, :, i));
    title(['Feature Map ', num2str(i)]);
    axis square;
    set(gca, 'XTick', [], 'YTick', []);
end

% Clear some room in memory
clear f i img pred qi