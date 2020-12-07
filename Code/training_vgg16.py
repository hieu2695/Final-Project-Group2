import os
import cv2
import pandas as pd
import numpy as np
import keras.backend as K
from skmultilearn.model_selection import iterative_train_test_split
from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.utils import plot_model
from skmultilearn.model_selection import iterative_train_test_split
from keras.models import load_model
import random
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain
from scipy.sparse import lil_matrix
from sklearn.metrics import fbeta_score


# ---------------------------------- Load Data ----------------------------------
# unzip train folder
PATH = os.getcwd()
file_name = 'human-protein-atlas-image-classification.zip'
#z = ZipFile(file_name)
count = 0
#z.extractall(path=PATH)

df_train = pd.read_csv('train.csv')

TRAIN_PATH = PATH + '/train/'
label_names = ['Nucleoplasm',
               'Nuclear membrane',
               'Nucleoli',
               'Nucleoli fibrillar center',
               'Nuclear speckles',
               'Nuclear bodies',
               'Endoplasmic reticulum',
               'Golgi apparatus',
               'Peroxisomes',
               'Endosomes',
               'Lysosomes',
               'Intermediate filaments',
               'Actin filaments',
               'Focal adhesion sites',
               'Microtubules',
               'Microtubule ends',
               'Cytokinetic bridge',
               'Mitotic spindle',
               'Microtubule organizing center',
               'Centrosome',
               'Lipid droplets',
               'Plasma membrane',
               'Cell junctions',
               'Mitochondria',
               'Aggresome',
               'Cytosol',
               'Cytoplasmic bodies',
               'Rods & rings']
# target variables



#------------------------ Exploratory Data Analysis -----------------------------
y = []
for ind, row in df_train.iterrows():
    labels = row['Target'].split()
    y.append([int(val) for val in labels])
y = np.array(y)


# Plot target labels
# create a single list of all labels in the train set
labels = []
for targets in y:
    for target in targets:
        labels.append(target)

# create dictionary of labels and their corresponding frequency
labels_count = {}
for element in labels:
    if element in labels_count:
        labels_count[element] += 1
    else:
        labels_count[element] = 1

category = [val for val in labels_count.keys()]
count = [val for val in labels_count.values()]

plt.figure()
plt.barh(category, count, color='red')
plt.xlabel('Number of Occurences')
plt.ylabel('Target Label')
plt.title('Target Labels for Image Data')
plt.yticks(np.arange(28), label_names)

plt.savefig('Target Labels.png')
print('current directory: ',os.getcwd())
plt.show()


# list of targets
y = []

# list of images
y_dict = {}
for ind, row in df_train.iterrows():
    y_dict[row['Id']]=row['Target']

num_label = []
for key in y_dict:
    y_dict[key] = y_dict[key].split(' ')
    num_label.append(len(y_dict[key]))
    y_dict[key] = np.array(y_dict[key])

plt.figure()
plt.hist(num_label)
plt.xlabel('Number of Labels')
plt.ylabel('Frequency')
plt.title('Number of Target Labels')
plt.savefig('Distribution of Target Labels.png')
plt.show()

#----------------------------------- import images and target labels --------------------

# set dimensions of images to 64 x 64
RESIZE_TO = 64
# list of targets
y = []

# list of images
images = []
count = 0
for filename in os.listdir(TRAIN_PATH):
    if str(filename)[-8:] == 'blue.png':
        # add to list of targets
        id = str(filename)[:-9]
        label = y_dict[id]
        y.append(label)

        # add to list of images
        img = cv2.resize(cv2.imread(os.path.join(TRAIN_PATH, filename)), (RESIZE_TO, RESIZE_TO))
        images.append(img)
        count = count + 1
        if count%10000 == 0:
            print(count)

y = np.array(y)
print(y.shape)
images = np.array(images)
print(y[:5])
print(images.shape)



# --------------------------- Splitting data into train and test set -----------------

SEED = 47
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=0.2)

x_train = x_train.reshape(len(x_train), RESIZE_TO, RESIZE_TO, 3)
x_test = x_test.reshape(len(x_test), RESIZE_TO, RESIZE_TO, 3)

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)


# ----------------------- Encode Target Label --------------------
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_test = mlb.transform(y_test)

print('y_train')
print(y_train[:5])
print(y_train.shape)
print('x_train')



# ------------------------------- CREATE AND COMPILE CNN MODEL ----------------------------

def pretrained_model():

    vgg_model = VGG16(input_shape=(RESIZE_TO, RESIZE_TO, 3), include_top=False)

    model = Sequential()
    for layer in vgg_model.layers:
        model.add(layer)

    for layer in model.layers:
        layer.trainable = False
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2, seed=SEED))
    model.add(Dense(28, activation='sigmoid'))
    print(model.summary())


    model.compile(loss='binary_crossentropy', metrics='accuracy', optimizer=Adam(lr=0.0001, decay=0.00005))
    return model



# ------------------------------------------- Train Model -------------------------------

N_EPOCHS = 25

# loading model
model = pretrained_model()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Image augmentation
datagen = ImageDataGenerator(
        rotation_range=10, # rotation
        width_shift_range=0.2, # horizontal shift
        height_shift_range=0.2, # vertical shift
        zoom_range=0.2, # zoom
        horizontal_flip=True, # horizontal flip
        brightness_range=[0.2,1.2],
        rescale=1./255,
        validation_split=0.2) # brightness


train_generator = datagen.flow(x_train, y_train, batch_size=64, subset='training')
validation_generator = datagen.flow(x_train, y_train, batch_size=64, subset='validation')


test_datagen = ImageDataGenerator(rescale=1/255.0)
test_generator = test_datagen.flow(x_test)



callback = ModelCheckpoint('model.hdf5', monitor='val_loss', save_best_only=True)
history = model.fit(train_generator, epochs=N_EPOCHS, validation_data=validation_generator, callbacks=[callback])


# Plotting loss and val_loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.figure()
plt.plot(epochs, loss, label='loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss and Val Loss vs Epochs')
plt.savefig('Loss and Val Loss vs Epochs.png')

plt.legend()

plt.show()

# make prediction
y_pred = model.predict(test_generator)

print(y_pred.shape)
print(y_test[:2])
print(y_pred[:2])



def encode_prediction(y):
    y_encoded = []
    for val in y:
        encoded_temp = []
        if val >0.3:
            encoded_temp.append(1)
        else:
            encoded_temp.append(0)
        y_encoded.append(encoded_temp)
    y_encoded = np.squeeze(y_encoded)

    return y_encoded
def list_predictions(y):
    y_encoded_list = []
    for val in y:
        encoded = encode_prediction(val)
        y_encoded_list.append(encoded)
    y_encoded_list = np.array(y_encoded_list)
    return y_encoded_list

y_pred = list_predictions(y_pred)
print('')
print(y_pred.shape)
print(y_test[:2])
print(y_pred[:2])

def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

