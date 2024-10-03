function blocks= split_jpg(img_dir)
BLOCK_WIDTH = 64;
BLOCK_HEIGHT = 64;
% Step 1: Read image
A = imread(img_dir);
% Step 2: Split image
A_rows = size(A,1);
A_cols = size(A,2);
% blocks_number = (A_rows/BLOCK_HEIGHT)*(A_cols/BLOCK_WIDTH);
blocks = cell(A_rows/BLOCK_HEIGHT, A_cols/BLOCK_WIDTH);

for i = 0:A_rows/BLOCK_HEIGHT-1
    for j = 0:A_cols/BLOCK_WIDTH-1
        blocks{i+1,j+1} = A(BLOCK_HEIGHT*i+1 : BLOCK_HEIGHT*(i+1), BLOCK_WIDTH*j+1 : BLOCK_WIDTH*(j+1), :);
    end
end


% Test Step (Redundant)
%size(blocks)
%blocks{1}
% if A(1:64,1:64,:) == blocks{1}
%    disp('dung')
% else
%    disp('sai')
% end
%
% imshow(blocks{1})