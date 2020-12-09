## Python scripts description:

### Note:
The data.zip needed to be downloaded and unzipped to data folder in the repo before running codes in this folder. Check the instruction in data_download.md in the repo to download data.

------------------------------------------------------------------------------------------------------------
The following scripts needed to be run first in the below order:
- "CreateDir.py": run this script 1st to create necessary directories.
- "load_data.py": load the image data and resize to 128x128 px with 4 channels (RGBY).
- "Data_Preprocessing.py": visualization, splitting data, oversampling minorities.

These scripts does not require correct order but it is recommended:
- "training_base.py": train the baseline model ResNet34 from scratch using the original data, all the layers are unfreezed.
- "training_resnet34_1.py": train the baseline model ResNet34 but using focal loss in the training process. The purpose is to understand the benefit of focal loss in addressing highly imblanced data.
- "training_resnet34_2.py": apply oversampled data, data augmentation, differential learning rate techniques to train the ResNet34 model using focal loss.
All the above models use ResNet34 architecture with 4 input channels.

- "training_resnet34_3.py": doing the same work as "training_resnet34_2", but keep the original input conv layer of ResNet34 (3 input channels), and add an 1x1 conv layer to the bottom of the architecture to convert 4 channel images to 3 channel images.
- "training_densenet121.py": use DenseNet architecure with mentioned techniques to classify images.

The following scripts needed to be run in the correct order:
- "training_resnet34_3_pre.py": freezed all layers of ResNet34 except the modified fc layer (top layer of the architecture), train the model to let fc layer initialized with optimal weights using augmented data (not including the original training set).
- "training_resnet34_3_full.py": unfreezed all layers, train the entire ResNet34 with learning rate 1e-3. (using original training set with augmented data).
- "training_resnet34_3_full_lr.py": train the entire ResNet34 with different learning rates for base layers, middle layers and head layers to let the model learn both generic information and specific, complex features (using original training set and a larger augmented data).

The following 2 scripts provide functions and classes to be used in other scripts:
- "Helper.py"
- "Helper1.py"
