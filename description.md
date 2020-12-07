## DATA SOURCE
https://www.kaggle.com/c/human-protein-atlas-image-classification/data

The data can be manually downloaded after accepting the competition rule.

Using Kaggle CLI command (requires Kaggle API token):

kaggle competitions download -c human-protein-atlas-image-classification

## Download from GCP

For convenience, I uploaded the data to my GCP so it can be downloaded using wget command.

Assuming that we are at the git repository Final-Project-Group2, else we need to move to the repo:

cd ~/Final-Project-Group2

At the git repo, download data via GCP using the command:

wget https://storage.googleapis.com/letranhieu-bucket-data/data.zip

Then, unzip the data.zip file:

unzip data.zip

The data will be unzipped to a folder named "data" containing "train" folder and "train.csv"

## Data Pre-process

The data contain 4 channels of images consist of Red, Green, Blue and Yellow (RGBY) which each of the channel is in the Gray scale images. The step to load the images is to read gray scale of each chennels and resize it to 128x128, then stack it together. 
For the targets of each image, the dataset is multi-labels dat with 28 classes totally in this dataset. So we create the dictionary for all classes then read the target file(train.csv) to get all targets for each image.

In the project, I split the data in to 3 set which are train(70%), validation(15%) and test set(15%). I create the figure to observe the imbalance of data and see which class is the minority class.
As the dataset is imbalance, I serach for the data that contain the minority classes which consist of Peroxisomes,Endosomes,Lysosomes, Microtubule ends, Mitotic spindle, Lipid droplets and Rods & rings in training set.
Then I use oversmapling along with data augmentation technique to make the data more balance.

# Modeling process

In this part, I used Keras for data augmentation purpose using ImageGenerator with rotation, width shift and height shift to generate more training samples with minority class. Pytorch is used to use for modeling purpose.





