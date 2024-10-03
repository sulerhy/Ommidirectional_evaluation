function blocks_info = PSNR_block(origins, distorts, foveation_x, foveation_y)
PSNR = zeros(size(origins));
for i=1:size(PSNR,1)
   for j=1:size(PSNR,2)
       % Step 1: Calculate Y U V
      [Y_origin, U_origin, V_origin] = RGB2yuv(origins{i,j});
      [Y_distorted, U_distored, V_distored] = RGB2yuv(distorts{i,j});
      
      % Step 2: Calculate PSNR (Y)
      MSE = 1/(size(Y_origin,1) * size(Y_origin,2))*sum((double(Y_distorted) -double(Y_origin)).^2, 'all'); 
      PSNR(i,j) = 10 * log10(255^2/MSE);
   end
end

% get default positions
positions=get_xy(8192, 4096, 64, 64, foveation_x, foveation_y);
blocks_info = zeros(size(origins,1), size(origins,2), 3);
blocks_info(:,:,1) = PSNR;
blocks_info(:,:,2:3) = positions;