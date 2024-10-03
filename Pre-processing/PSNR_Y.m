function PSNR = PSNR_Y(Y_origin, Y_distorted)

% Step 0: Check Y_origin and Y_distorted has same size
if size(Y_origin) ~= size(Y_distorted)
    disp("Two matrixes do not have same size, please check again")
end
% Step 1: Calculate MSE
MSE = 1/(size(Y_origin,1) * size(Y_origin,2))*sum((double(Y_distorted) -double(Y_origin)).^2, 'all'); 
% Step 2: Calculate PSNR (Y)
PSNR = 10 * log10(255^2/MSE);