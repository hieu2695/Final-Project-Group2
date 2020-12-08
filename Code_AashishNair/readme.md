Description of the data set:

The dataset consists of over 120,000 images of protein patterns. While that is an astonishingly high number, it does not represent the number of samples provided. Each sample in this dataset is represented by 4 images, with each image having the colors of blue, green, red, and yellow. The dataset contains a total of 28 different labels. It should be noted that these images are multilabel, meaning that each image may contain more than 1 label. While most images have only 1 label, there are a considerable number of images that have 2 and 3 labels. There are even images, albeit a few, that have 4 labels. 

 Furthermore, the distribution of the labels in this dataset are not evenly balanced. In fact, images with the labels “Nucleoplasm” and “Cytosol” comprise of over 80% of the images provided. This massive imbalance of data must be accounted for when developing the deep neural network. In terms of dimensions, the images are 512 x 512, though they will be resized prior to training the model. Since any size equal to or above 128 x 128 pixels registers an error from memory shortage, the images will be resized to 64 x 64 pixels.

Description of the deep learning network and training algorithm:

The neural network used in this project will a convolutional neural network (CNN) since they excel in applications like image classification. Instead of building a CNN from scratch, it would be beneficial to implement transfer learning and take advantage of a tried and trusted pre-trained model. The chosen network for this project is the VGG16 model, which contains numerous convolutional layers and pooling layers designed to tackle problems just like this one. Once the vgg16 model would be modified with the Keras framework to suit the input from the protein images, it will be trained to distinguish between protein patterns effectively and accurately. The layout of the altered vgg16 model can be seen below:

