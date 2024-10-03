<h3> # Omnidirectional_Images_Assessment </h3>
<h4> Omnidirectional Images Assessment </h4>
<div> In this project, I start to research on the Omnidirectional (ˌämniˌdiˈrekSHənl - 3D) Images and how to assess them by Machine Learning and make them as the same result as human perspective </div>

There are two phases: <br />
**First phase**: Pre-processing: Image spliting, PSNR calculating, position of each patch calculate, etc.<br />
**Second phase**: Apply deep learning to predict MOS of image base on data has been processed on the first phase


**Result**: Below is the best training result based on the dataset of 512 Omnidirectional images.

The Pearson Correlation at **0.9757**, demonstrates the model predicted results correlate to the human manual evaluation (ground truth) at **97.57%**

![best_training.png](best_training.png)