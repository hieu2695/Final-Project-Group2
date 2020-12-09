## About this folder:
Contain the best model that we obtained in this project.

## Python scripts description:

### Note:
The data.zip needed to be downloaded and unzipped to data folder in the repo before running codes in this folder. Check the instruction in data_download.md in the repo to download data.

------------------------------------------------------------------------------------------------------------
The following scripts needed to be run correct order:
Preprocessing:
- "CreateDir.py": run this script 1st to create necessary directories.
- "load_data.py": load the image data and resize to 128x128 px with 4 channels (RGBY).
- "Data_Preprocessing.py": visualization, splitting data, oversampling minorities.

Modeling:
- "training_resnet34_3_pre.py": freezed all layers of ResNet34 except the modified fc layer (top layer of the architecture), train the model to let fc layer initialized with optimal weights using augmented data (not including the original training set).
- "training_resnet34_3_full.py": unfreezed all layers, train the entire ResNet34 with learning rate 1e-3. (using original training set with augmented data).
- "training_resnet34_3_full_lr.py": train the entire ResNet34 with different learning rates for base layers, middle layers and head layers to let the model learn both generic information and specific, complex features (using the original training set and a larger augmented data).

The following 2 scripts provide functions and classes to be used in other scripts:
- "Helper.py"
- "Helper1.py"
