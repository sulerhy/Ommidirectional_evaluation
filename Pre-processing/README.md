<h3> # Omnidirectional_Images_Assessment </h3>
<h4> Omnidirectional Images Assessment </h4>
<div> In this phase, we first convert RGB images into YUV. After that, we split the image into a lot of patches with size of 64x64x3. Calculate PSNR and distance to foveation point of each patch. </div>
<div>Input: 4096x8192x3 (RGB, 8K Omnidirectional images)</div>
<div>Output: 64x128x3 (PSNR, distance to foveation point as x-axis, y-axis)</div>
<div>Usage <br>
1. Copy Files into OriginalImage and TestImage folders <br>
2. Run main.m <br>
The result will be saved into training_set.mat file which includes: <br>
  struct with fields: <br>

       MOS: [512×1 double] <br>
    result: [512×64×128×3 double] <br>


</div>
