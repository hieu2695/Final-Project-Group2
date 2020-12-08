## Python scripts description:

### Note:
The data.zip needed to be downloaded and unzipped to data folder in the repo before running codes in this folder.

The following scripts needed to be run first in the below order:
- "CreateDir.py": run this script 1st to create necessary directories.
- "load_data.py": load the image data and resize to 128x128 px with 4 channels (RGBY).
- "Data_Preprocessing.py": visualization, splitting data, oversampling minorities.

These scripts does not require correct order but it is recommended:
- "training_base.py": train the baseline model ResNet34 from scratch using the original data, all the layers are unfreezed.
- "training_resnet34_1.py": train the baseline model ResNet34 but using focal loss in the training process. The purpose is to understand the benefit of focal loss.
