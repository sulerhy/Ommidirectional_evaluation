function [Y,U,V]=RGB2yuv(RGB)
R = RGB(:,:,1); 
G = RGB(:,:,2); 
B = RGB(:,:,3);
[Y,U,V]=rgb2yuv(R,G,B,'YUV420_8','BT601_f');

% imshow(Y);
% [Y1, U1, V1] = yuv_import('input_yuv.yuv',[64 64],1);
% PSNR_Y(Y, Y1{1})
% if (Y+3)==Y1{1}
%    disp('Bang nhau')
% else
%    disp('khong bang nhau')
% end
%double(Y1{1})-double(Y)
% image_show(Y1{1},256,1,'Y component');