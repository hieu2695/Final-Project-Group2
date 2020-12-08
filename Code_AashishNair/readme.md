Description of the data set:

The dataset consists of over 120,000 images of protein patterns. While that is an astonishingly high number, it does not represent the number of samples provided. Each sample in this dataset is represented by 4 images, with each image having the colors of blue, green, red, and yellow. The dataset contains a total of 28 different labels. It should be noted that these images are multilabel, meaning that each image may contain more than 1 label. While most images have only 1 label, there are a considerable number of images that have 2 and 3 labels. There are even images, albeit a few, that have 4 labels. 

Furthermore, the distribution of the labels in this dataset are not evenly balanced. In fact, images with the labels “Nucleoplasm” and “Cytosol” comprise of over 80% of the images provided. This massive imbalance of data must be accounted for when developing the deep neural network. In terms of dimensions, the images are 512 x 512, though they will be resized prior to training the model. Since any size equal to or above 128 x 128 pixels registers an error from memory shortage, the images will be resized to 64 x 64 pixels.


Description of the deep learning network and training algorithm:

The neural network used in this project will a convolutional neural network (CNN) since they excel in applications like image classification. Instead of building a CNN from scratch, it would be beneficial to implement transfer learning and take advantage of a tried and trusted pre-trained model. The chosen network for this project is the VGG16 model, which contains numerous convolutional layers and pooling layers designed to tackle problems just like this one. Once the vgg16 model would be modified with the Keras framework to suit the input from the protein images, it will be trained to distinguish between protein patterns effectively and accurately. 


Experimental setup:

Firstly, the data will be split into training and test sets, with the test set being used to evaluate the trained network. The training set will once again be split into a training set and validation set. The training set, and validation set, and test set make up 64%, 16%, and 20%  of the available data, respectively. Originally, it was intended to for K cross validation (or stratified K cross validation) to be utilized to split the training data into smaller subsets to ensure better results. However, this method proved to be severely time consuming and provided minimal marginal benefits. Therefore, this approach was soon abandoned. 

The performance of the model will be judged based on the loss function and the f1-score. The loss function and f1-score metric chosen are the binary cross-entropy and the macro f1-score as they are most suited for a multilabel classification problem such as this one. 

The key training parameters under scrutiny during the development of the neural network are the learning rate, batch size, optimizer, dropout value, and activation function. The ideal value for these parameters were determined by experimenting with different combinations of values and seeing which one yielded the best results. 

Through some experimentation, the ideal parameters for this model were determined. The assigned values for the learning rate, batch size, optimizer, dropout value, and activation function for this model are 1e-4, 64, 'Adam', 0.2, and 'relu', respectively. 


Results:

This VGG16 model registered a validation loss of 0.17 and a validation f1-score of 0.14. When tested on the test set, the registered f1-score was 0.11. Clearly, the efforts made to address the data imbalance the the prone to overfitting were not sufficient. 


How to run the code:

First, run the code in the "download_data.py" file. This will download and unzip a folder called "data" that will contain all necessary files for training the model. Then, run the "training_vgg16.py" file to load the images and csv files in the data folder and use it to train the VGG16 model. 


