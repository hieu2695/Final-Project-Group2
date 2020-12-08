import torch
import torch.nn as nn
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.metrics import f1_score
from torchvision import models
import copy
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DATA_DIR = os.getcwd()
n_classes = 28

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# load the data
# split data into the training set and testing set
x_train, y_train = np.load("../processed_data/splitted_data/x_train.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_train.npy", allow_pickle=True)
x_train_over, y_train_over = np.load("../processed_data/splitted_data/x_train_over.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_train_over.npy", allow_pickle=True)
x_val, y_val = np.load("../processed_data/splitted_data/x_val.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_val.npy", allow_pickle=True)
x_test, y_test = np.load("../processed_data/splitted_data/x_test.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_test.npy", allow_pickle=True)

#%%-------------------- Data Augmentation -----------------
def Data_Augmment_Generate(imgs,targets_set,n):
# imgs is the list of images array to generate more image
# n is the no. of iterate to generate
# generate is the empty list to save the generate file
    Image_Gen = ImageDataGenerator(rotation_range=90, width_shift_range=[-2,2], height_shift_range=[-0.03, 0.03])
    generate = []
    targets = []
    for i in range(len(imgs)):
        image = imgs[i]
        image = expand_dims(image, 0)
        it = Image_Gen.flow(image, batch_size=1, seed=SEED)
        for j in range (n):
            batch = it.next()
            # convert to integers
            temp = batch[0].astype('uint8')
            generate.append(temp)
            targets.append(targets_set[i])

    generate = np.asarray(generate)
    targets = np.asarray(targets)
    return generate, targets
n = 3
x_aug, y_aug = Data_Augmment_Generate(x_train_over,y_train_over,n)
np.save("x_train_genearated.npy", x_aug); np.save("y_train_genearated.npy", y_aug)

# Use this code to load then augment data after already generated the data
#x_aug, y_aug = np.load("x_train_genearated.npy"), np.load("y_train_genearated.npy")

# imprt from cv2 image: (W x H x C) change to torch image: (C x H x W)

x_train = x_train.transpose((0, 3, 1, 2))
x_aug = x_aug.transpose((0, 3, 1, 2))
x_val = x_val.transpose((0, 3, 1, 2))
x_test = x_test.transpose((0, 3, 1, 2))

# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape)
# print(x_test.shape, y_test.shape)

x_train, y_train = torch.from_numpy(np.asarray(x_train)), torch.from_numpy(np.asarray(y_train))
x_aug, y_aug = torch.from_numpy(np.asarray(x_aug)), torch.from_numpy(np.asarray(y_aug))
x_val, y_val = torch.from_numpy(np.asarray(x_val)), torch.from_numpy(np.asarray(y_val))
x_test, y_test = torch.from_numpy(np.asarray(x_test)), torch.from_numpy(np.asarray(y_test))

raw_train_set = TensorDataset(x_train,y_train)
aug_set = TensorDataset(x_aug,y_aug)
val_set = TensorDataset(x_val,y_val)
test_set = TensorDataset(x_test,y_test)

train_set = torch.utils.data.ConcatDataset([raw_train_set, aug_set])

trainloader = torch.utils.data.DataLoader(train_set, batch_size = 200, shuffle=True)
valloader = torch.utils.data.DataLoader(val_set, batch_size = 200, shuffle=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size = 200, shuffle=True)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-4
N_EPOCHS = 50
DROPOUT = 0.05
R = 4 # Input size
a_size = n_classes # Output size
early_stop = 5

# %% ----------------------------------- Model CNN --------------------------------------------------------------
model = models.resnet152(pretrained=True)

# changing the 1st convolution layer input channels to 4
weight = model.conv1.weight.clone() # clone a copy of weight for conv1
model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

# copy back to weights to conv1
with torch.no_grad():
    model.conv1.weight[:, :3] = weight
    model.conv1.weight[:, 3] = model.conv1.weight[:, 0]

num_ftrs = model.fc.in_features

model.fc =  nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Linear(256, n_classes)
    )
for param in model.parameters():
    param.requires_grad = True

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = model.type('torch.FloatTensor').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop")
train_loss = []
val_loss = []
running_f1_macro = []
running_f1_samples = []
for epoch in range(N_EPOCHS):
    train_running_loss = 0
    model.train()
    print("Epoch : ", epoch+1)
    for i, data in enumerate(trainloader,0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss_temp = loss.item() * len(inputs)
        train_running_loss += train_loss_temp
    # track the mean loss for each epoch
    train_loss_temp = train_running_loss/len(trainloader.sampler)
    train_loss.append(train_loss_temp)
    print("Training Loss ---> " + str(train_loss_temp))

    # Check the loss and F-1 score in valid
    with torch.no_grad():
        val_preds = []
        val_targets = []
        val_running_loss = 0
        for i, data in enumerate(valloader, 0):
            inputs, val_labels = data
            inputs, val_labels = inputs.to(device), val_labels.to(device)
            inputs, val_labels = Variable(inputs), Variable(val_labels)
            y_val_pred = model(inputs)
            y_val_pred = torch.sigmoid(y_val_pred)
            y_val_pred = torch.round(y_val_pred)
            loss = criterion(y_val_pred, val_labels)
            val_loss_temp = loss.item()* len(inputs)
            val_running_loss += val_loss_temp

            true_targets = val_labels.cpu().detach().numpy()
            for true_target in true_targets:
                 val_targets.append(true_target)

            y_val_pred = y_val_pred.cpu().detach().numpy()
            for pred in y_val_pred:
                val_preds.append(pred)

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        F1_macro = f1_score(val_targets, val_preds , average='macro')
        F1_sample = f1_score(val_targets, val_preds , average='samples')
        running_f1_macro.append(F1_macro)
        running_f1_samples.append(F1_sample)

    # track the loss for each epoch
    val_loss_temp = val_running_loss/len(valloader.sampler)
    val_loss.append(val_loss_temp)
    print("Validation Loss ---> " + str(val_loss_temp))
    print("F-1 macro score on validation set ---> " + str(F1_macro))
    print("F-1 sample score on validation set ---> " + str(F1_sample))

    # save model with better F1-score and add early stop if the model is not improve
    if epoch == 0:
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, "best_model.pt")
        best_macro_F1 = F1_macro
        best_sample_F1 = F1_sample
        count = 0

    else:
        if F1_macro > best_macro_F1:
            best_macro_F1 = F1_macro
            best_sample_F1 = F1_sample
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "Resnet50_model_2.pt")
            count = 0
        else :
            count += 1

    if count > early_stop :
        break
# %% -------------------------------------- Testing Loop ----------------------------------------------------------

model.eval()  # change to evaluate mode
test_loss = []
test_preds = []
test_targets = []
test_running_loss = 0
for i, data in enumerate(testloader, 0):
    inputs, test_labels = data
    inputs, test_labels = inputs.to(device), test_labels.to(device)
    inputs, test_labels = Variable(inputs), Variable(test_labels)
    y_test_pred = model(inputs)
    y_test_pred = torch.sigmoid(y_test_pred)
    y_test_pred = torch.round(y_test_pred)
    loss = criterion(y_test_pred, test_labels)
    test_loss_temp = loss.item() * len(inputs)
    test_running_loss += test_loss_temp

    true_targets = test_labels.cpu().detach().numpy()
    for true_target in true_targets:
        test_targets.append(true_target)

    y_test_pred = y_test_pred.cpu().detach().numpy()
    for pred in y_test_pred:
        test_preds.append(pred)

test_loss = test_running_loss/len(testloader.sampler)
test_preds = np.array(test_preds)
test_targets = np.array(test_targets)
test_F1_macro = f1_score(test_targets, test_preds, average='macro')
test_F1_sample = f1_score(test_targets, test_preds, average='samples')

print('Testing set :')
print("Testing Loss ---> " + str(test_loss))
print("F-1 macro score on validation set ---> " + str(test_F1_macro))
print("F-1 sample score on validation set ---> " + str(test_F1_sample))
print(model)