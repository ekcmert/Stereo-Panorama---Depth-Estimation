clear all ; close all ; clc;

I1 = imread("11080086_l.PNG");
I2 = imread("11080086_r.PNG");

% Convert to grayscale.
I1gray = rgb2gray(I1);
I2gray = rgb2gray(I2);
figure;
imshowpair(I1, I2,"montage");
title("I1 (left); I2 (right)");
figure; 
imshow(stereoAnaglyph(I1,I2));
title("Composite Image (Red - Left Image, Cyan - Right Image)");
blobs1 = detectSURFFeatures(I1gray,MetricThreshold=2000);
blobs2 = detectSURFFeatures(I2gray,MetricThreshold=2000);
figure; 
imshow(I1);
hold on;
plot(selectStrongest(blobs1,50));
title("Fifty Strongest SURF Features In I1");
figure; 
imshow(I2); 
hold on;
plot(selectStrongest(blobs2,50));
title("Fifty Strongest SURF Features In I2");
[features1,validBlobs1] = extractFeatures(I1gray,blobs1);
[features2,validBlobs2] = extractFeatures(I2gray,blobs2);
indexPairs = matchFeatures(features1,features2,Metric="SAD", MatchThreshold=50);
matchedPoints1 = validBlobs1(indexPairs(:,1),:);
matchedPoints2 = validBlobs2(indexPairs(:,2),:);
figure; 
showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2);
legend("Putatively Matched Points In I1","Putatively Matched Points In I2");
[fMatrix, epipolarInliers, status] = estimateFundamentalMatrix(matchedPoints1,matchedPoints2,Method="RANSAC",NumTrials=10000,DistanceThreshold=0.01,Confidence=99.99);
  
if status ~= 0 || isEpipoleInImage(fMatrix,size(I1)) ...
  || isEpipoleInImage(fMatrix',size(I2))
  error(["Not enough matching points were found or "...
         "the epipoles are inside the images. Inspect "...
         "and improve the quality of detected features ",...
         "and images."]);
end

inlierPoints1 = matchedPoints1(epipolarInliers, :);
inlierPoints2 = matchedPoints2(epipolarInliers, :);

figure;
showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2)
legend("Inlier Points In I1","Inlier Points In I2");
[tform1, tform2] = estimateStereoRectification(fMatrix, ...
  inlierPoints1.Location,inlierPoints2.Location,size(I2));
[I1Rect, I2Rect] = rectifyStereoImages(I1,I2,tform1,tform2);
figure;
imshow(stereoAnaglyph(I1Rect,I2Rect));
title("Rectified Stereo Images (Red - Left Image, Cyan - Right Image)");
figure
subplot(1,2,1)
imshow((I1Rect))
title("I1");
subplot(1,2,2)
imshow((I2Rect))
title("I2");



img_right = I1Rect;
img_right = rgb2gray(img_right);
img_right = double(img_right);
[row, column,ch] = size (img_right);
    %img_right = imresize(img_right,[4000,4000])
img_left = I2Rect;
img_left = rgb2gray(img_left);
img_left = double(img_left);
    %img_left = imresize(img_left,[4000,4000])

[row, column,ch] = size (img_left);
    
k= 5;
    omega = 8;
    offset = omega + k;


    img_L = padarray(img_left,[offset offset],'both');
    img_R = padarray(img_right,[offset offset],'both');
    [ydim,xdim]=size(img_L);




    for xL = offset+1:1:xdim-offset-1
        for yL = offset+1:1:ydim-offset-1
        dist = [];
        subL =img_L(yL-k:yL+k,xL-k:xL+k);
            for xR = xL:-1:xL-omega
                subR = img_R(yL-k:yL+k,xR-k:xR+k);
                C = sum(sum(-1.*(subL-subR).^2));
    
                dist=[dist;xL-xR C];
            end
        ind = find(dist(:,2) == max(dist(:,2)));
        d = dist(ind(1),1);
        disparity(yL,xL)=d;
        %disparity=disparity(yL,xL);
        end 
    end

imshow(stereoAnaglyph(uint8(img_L),uint8(img_R)));
% Show disparity map with colorbar
figure; imagesc(uint8(disparity)); colormap turbo; colorbar


