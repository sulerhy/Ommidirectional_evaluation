In this phase, we implement CNN model to predict MOS of omnidirectional images from dataset which has been processed in phase 1 (pre-processing).

Usage:
Step 1: Run init_dataset.py to generate dataset from data/training_set.mat (from phase 1). Dataset will be saved into dataset folder
Step 2: Run basic_model to training model. Best weight and model will be saved into models folder. Result will be saved into result folder
Step 3: Run visualize_results.py to visualize result and see how good model perform
Result will be show in scatter graph and calculate PCC and PMSE also.
