import torch
import torch.nn as nn
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.metrics import f1_score
import copy

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 500
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
x_val, y_val = np.load("../processed_data/splitted_data/x_val.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_val.npy", allow_pickle=True)
x_test, y_test = np.load("../processed_data/splitted_data/x_test.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_test.npy", allow_pickle=True)

# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape)
# print(x_test.shape, y_test.shape)

# imprt from cv2 image: (W x H x C) change to torch image: (C x H x W)
x_train = x_train.transpose((0, 3, 1, 2))
x_val = x_val.transpose((0, 3, 1, 2))
x_test = x_test.transpose((0, 3, 1, 2))

# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape)
# print(x_test.shape, y_test.shape)

x_train, y_train = torch.from_numpy(np.asarray(x_train)), torch.from_numpy(np.asarray(y_train))
x_val, y_val = torch.from_numpy(np.asarray(x_val)), torch.from_numpy(np.asarray(y_val))
x_test, y_test = torch.from_numpy(np.asarray(x_test)), torch.from_numpy(np.asarray(y_test))

# x_train,y_train = x_train.to(device), y_train.to(device)
# x_val,y_val = x_val.to(device), y_val.to(device)
# x_test,y_test = x_test.to(device), y_test.to(device)

train_set = TensorDataset(x_train,y_train)
val_set = TensorDataset(x_val,y_val)
test_set = TensorDataset(x_test,y_test)

trainloader = torch.utils.data.DataLoader(train_set, batch_size = 200, shuffle=True)
valloader = torch.utils.data.DataLoader(val_set, batch_size = 200, shuffle=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size = 200, shuffle=True)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-4
N_EPOCHS = 150
DROPOUT = 0.05
R = 4 # Input size
a_size = n_classes # Output size
early_stop = 10

# %% ----------------------------------- Model CNN --------------------------------------------------------------

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(4, 16, (3, 3)) # Output dim = (128-3)/1 + 1 = 126
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2)) # Output dim = (126-2)/2 + 1 = 63
        self.conv2 = nn.Conv2d(16, 32, (3, 3)) # Output dim = (63-3)/1 + 1 = 61
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2)) # Output dim = (61 - 2)/2 + 1 = 30
        self.conv3 = nn.Conv2d(32, 64, (3, 3)) # Output dim = (30-3)/1 + 1 = 28
        self.convnorm3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d((2, 2)) # Output dim = (28 - 2)/2 + 1 = 14
        self.conv4 = nn.Conv2d(64, 64, (3, 3))# Output dim = (14-3)/1 + 1 = 12
        self.pool4 = nn.MaxPool2d((2, 2)) # Output dim = (12 - 2)/2 + 1 = 6
        self.linear1 = nn.Linear(64 * 6 * 6, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 28)
        self.act = torch.relu
        self.drop = nn.Dropout(DROPOUT)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop(self.pool1(self.convnorm1(self.act(self.conv1(x)))))
        x = self.drop(self.pool2(self.convnorm2(self.act(self.conv2(x)))))
        x = self.drop(self.pool3(self.convnorm3(self.act(self.conv3(x)))))
        x = self.drop(self.pool4(self.convnorm3(self.act(self.conv4(x)))))
        x = self.act(self.linear1(x.view(len(x), -1)))
        x = self.act(self.linear2(self.drop(x)))
        x = self.linear3(self.drop(x))
        x = self.sigmoid(x)
        return x

# %% -------------------------------------- Training Prep ----------------------------------------------------------

model = CNN().type('torch.FloatTensor').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

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
            torch.save(best_model_wts, "Custom_model.pt")
            count = 0
        else :
            count += 1

    if count > early_stop :
        break
# %% -------------------------------------- Testing Loop ----------------------------------------------------------
model.load_state_dict(torch.load("custom_model.pt"))
model.eval()  # change to evaluate mode
test_preds = []
test_targets = []
test_running_loss = 0
for i, data in enumerate(testloader, 0):
    inputs, test_labels = data
    inputs, test_labels = inputs.to(device), test_labels.to(device)
    inputs, test_labels = Variable(inputs), Variable(test_labels)
    y_test_pred = model(inputs)
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
