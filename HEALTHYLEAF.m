clc; clear; close all;

%% Step 1: Select Image File
[file, path] = uigetfile({'*.jpg;*.png;*.bmp'}, 'Select a Leaf Image');
if isequal(file, 0)
    disp('User cancelled the image selection.');
    return;
end

img = imread(fullfile(path, file));
figure('Name', ['Processing: ', file], 'NumberTitle', 'off');

% Step 2: Show Original Image
subplot(2, 3, 1);
imshow(img); title('Original Image');

% Step 3: Convert to Grayscale
if size(img, 3) == 3
    gray_img = rgb2gray(img);
else
    gray_img = img;
end
subplot(2, 3, 2);
imshow(gray_img); title('Grayscale');

% Step 4: Binary Mask of Leaf Area
bw_img = imbinarize(gray_img, graythresh(gray_img));
bw_img = imfill(bw_img, 'holes');
leaf_area = sum(bw_img(:));
subplot(2, 3, 3);
imshow(bw_img); title('Leaf Mask');

% Step 5: Disease Area Detection
disease_mask = imsubtract(im2double(img(:,:,1)), im2double(img(:,:,2)));
disease_mask = imbinarize(disease_mask, 0.15);
disease_mask = bwareaopen(disease_mask, 100);
subplot(2, 3, 4);
imshow(disease_mask); title('Disease Areas');

% Step 6: Disease Area Calculation
affected_area = sum(disease_mask(:));
concentration = (affected_area / leaf_area) * 100;

% Step 7: Feature Extraction
avg_intensity = mean(gray_img(:));
contrast = std(double(gray_img(:)));
smoothness = 1 / (1 + contrast);
third_moment = skewness(double(gray_img(:)));
uniformity = sum(imhist(gray_img).^2);
entropy_val = entropy(gray_img);

features = [avg_intensity, contrast, smoothness, third_moment, uniformity, entropy_val];

% Step 8: Feature Database
diseases = {'Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut'};
feature_db = [
    186.4631, 55.3098, 0.04518, -1.59633, 0.0768, 1.531915; % Leaf Blight
    182.8573, 49.9941, 0.09383, -0.5582, 0.0804, 5.4702;    % Brown Spot
    176.0134, 55.3379, 0.0469, -0.6033, 0.0811, 5.2226      % Leaf Smut
];

% Step 9: Classification Based on Euclidean Distance
distances = vecnorm(feature_db - features, 2, 2);
[min_dist, idx] = min(distances);

% Step 10: Smart Thresholding
% Instead of fixed 60, compute max tolerable distance dynamically
avg_db_distance = mean(vecnorm(feature_db - mean(feature_db), 2, 2));
tolerance_factor = 1.5;
threshold = avg_db_distance * tolerance_factor;

if min_dist > threshold
    classified_disease = 'Healthy Leaf';
else
    classified_disease = diseases{idx};
end

% Step 11: Show Final Result
subplot(2, 3, [5 6]);
imshow(img);
title(['Disease: ', classified_disease, ' | Concentration: ', num2str(concentration, '%.2f'), '%']);

% Step 12: Output Info
msg = sprintf(['Image: %s\nDetected Disease: %s\nConcentration: %.2f%%\nDistance: %.2f (Threshold: %.2f)'], ...
    file, classified_disease, concentration, min_dist, threshold);

disp(msg);
helpdlg(msg, 'Detection Result');
