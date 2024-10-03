A = imread('input_block.jpg');
R = A(:,:,1); 
G = A(:,:,2); 
B = A(:,:,3);
[Y,U,V]=rgb2yuv(R,G,B,'YUV420_8','BT601_l');


[Y1, U1, V1] = yuv_import('input_yuv.yuv',[64 64],2);

if (Y+3)==Y1{1}
    disp('Bang nhau')
else
    disp('khong bang nhau')
end
double(Y1{1})-double(Y)
% image_show(Y1{1},256,1,'Y component');