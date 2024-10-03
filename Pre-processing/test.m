org_img = strcat('OriginalImage\I1.jpg');
% test_img = strcat('TestImage\I1_blur_2.5_7.5_0_8.jpg');
test_img = strcat('TestImage\I1_blur_2.5_7.5_0_12.jpg');
A = imread(org_img);
A_blocks = split_jpg(org_img);
B = imread(test_img);
B_blocks = split_jpg(test_img);

[Y_origin, U_origin, V_origin] = RGB2yuv(cell2mat(A_blocks(35,64)));
[Y_distorted, U_distored, V_distored] = RGB2yuv(cell2mat(B_blocks(35,64)));
MSE = 1/(size(Y_origin,1) * size(Y_origin,2))*sum((double(Y_distorted) -double(Y_origin)).^2, 'all')
PSNR = 10 * log10(255^2/MSE)