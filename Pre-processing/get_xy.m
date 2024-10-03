function A = get_xy(x_origins,y_origins, x_block, y_block, fove_x, fove_y)
% x_origins: width of input image
% y_origins: height of input image
% x_block: width of a block
% y_block: height of a block
% fove_x: foveation point by x coordinate
% fove_y: foveation point by y coordinate

height = y_origins/y_block;
width = x_origins/x_block;
A = zeros(height,width,2);
for i=1:height
	for j=1:width
    	block_center_x = (j-0.5)*x_block;
    	block_center_y = (i-0.5)*y_block;
    	x_distance(i,j) = abs(block_center_x - fove_x)/x_origins;
    	y_distance(i,j) = abs(block_center_y - fove_y)/y_origins;
	end
end

A(:,:,1) = x_distance;
A(:,:,2) = y_distance;

% height, width imaging
%  |--------width---------|
%  |----------------------|
% height----------------height
%  |----------------------|
%  |--------width---------|