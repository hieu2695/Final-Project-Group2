## Data Pre-process

The data contain 4 channels of images consist of Red, Green, Blue and Yellow (RGBY) which each of the channel is in the Gray scale images. The step to load the images is to read gray scale of each chennels and resize it to 128x128, then stack it together. 
For the targets of each image, the dataset is multi-labels dat with 28 classes totally in this dataset. So we create the dictionary for all classes then read the target file(train.csv) to get all targets for each image.

In the project, I split the data in to 3 set which are train(70%), validation(15%) and test set(15%). I create the figure to observe the imbalance of data and see which class is the minority class.
As the dataset is imbalance, I serach for the data that contain the minority classes which consist of Peroxisomes,Endosomes,Lysosomes, Microtubule ends, Mitotic spindle, Lipid droplets and Rods & rings in training set.
Then I use oversmapling along with data augmentation technique to make the data more balance.

## Modeling process

Pytorch is used to use for modeling purpose. Starting with the created model, I used 4 2DConvolute layers starting with 16 channels then 32, 64 and 128 then respectively with Batch Normalization and Dropout. Then I used 3 linear layers to flatten the output from Convolute layers. The activation function is Relu and then using BCEWithLogitsLoss for loss function. I created the save point and early stop by F1-score macro and samples on validation set then final evalution the model on test set.

Result for created model without data augmentation :
Validation set

Test set :

Next, I used pre-trained model Resnet50 by chaning the first input layer to be 4 channel for using in our dataset and the output to be 28.
/n 
Result for Resnet50 model without data augmentation :
Validation set

Test set :

I used Keras for data augmentation purpose using ImageGenerator with rotation, width shift and height shift to generate more training samples with minority class.

Result for Resnet50 model with data augmentation :

Validation set :

Validation Loss ---> 0.6911363989357057

F-1 macro score on validation set ---> 0.5304892918144725

F-1 sample score on validation set ---> 0.5794594463801781

Test set :

Testing Loss ---> 0.6910124851647347

F-1 macro score on validation set ---> 0.4893733114790944

F-1 sample score on validation set ---> 0.5773960023263643

The last one, I tried Resnet152 model with data augmentation.

Result for Resnet152 model with data augmentation :

Validation set :

Validation Loss ---> 0.6910018179474807

F-1 macro score on validation set ---> 0.5699673836926441

F-1 sample score on validation set ---> 0.6019212156407279

Test set :

Testing Loss ---> 0.6912819459884305

F-1 macro score on validation set ---> 0.5132726667051809

F-1 sample score on validation set ---> 0.5981695185037803





