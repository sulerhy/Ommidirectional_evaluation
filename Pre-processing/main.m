addpath("YUV");
% foveation point coordinate
foveation_x=6144;
foveation_y=3072;

number_test_imgs = 512;
number_tests_a_img = 32;
% MOS of images 
MOS = xlsread('SubjectiveMOS.xlsx', 'Sheet1', strcat('H2:H',int2str(number_test_imgs + 1)));
% original images
[~,origin_names] = xlsread('SubjectiveMOS.xlsx', 'Sheet1',strcat('D2:D',int2str(number_test_imgs + 1)));
% test images
[~,test_names] = xlsread('SubjectiveMOS.xlsx', 'Sheet1', strcat('G2:G',int2str(number_test_imgs + 1)));

% result will be the matrix with size of (number_images, height_blocks, width_blocks, 3)
% ex: (512, 64, 128, 3)
% 3: PSNR, position by x axis, position by y axis
result = zeros(number_test_imgs, 64, 128, 3);
origins = zeros(64,128,3);
distorts = zeros(64,128,3);
file_output = strcat('training-set_', int2str(foveation_x), '_', int2str(foveation_y), '.mat');
for i = 1:number_test_imgs
    disp(strcat('Process: ',int2str(i), '/', int2str(number_test_imgs)));
    % PSNR by Y and positions of all blocks
    org_img = strcat('OriginalImage\',origin_names(i,1),'.jpg');
    test_img = strcat('TestImage\',test_names(i,1));
    if mod(i-1,32)==0
        % split original image into blocks of 64x64x3
        origins = split_jpg(org_img{1});
    end
    % split distorted image into blocks of 64x64x3 
    distorts = split_jpg(test_img{1});
    % calculate PSNR and positions
    result(i,:,:,:) = PSNR_block(origins, distorts, foveation_x, foveation_y);
    if mod(i,32)==0
        save(file_output, 'result', 'MOS');
        disp("Data saved.")
    end
end
disp("Get trainning set: Finished")
save(file_output, 'result', 'MOS'); 
disp("Save data: Done")
