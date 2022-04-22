clear;
clc;
close all;
format compact;

load('color-info.mat');
while true
    color = input('Color? ');
    rgb = input('RGB? ');
    
    if (strcmp(color, 'end'))
        break
    elseif (~any(strcmp(color_names, color)))
        color_names{end+1} = color;
        color_rgb(end+1, :) = rgb;
    end
end

clear color rgb
save('color-info.mat');