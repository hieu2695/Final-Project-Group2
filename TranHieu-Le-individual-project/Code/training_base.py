# ==================================== BASELINE RESNET34 =============================================================
#%% ------------------------------------ Import Lib ----------------------------------------------------------------
import torch
import torch.nn as nn
import os
import random
import numpy as np
from Helper import train_baseline_model, DataAug
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# number of labels
n_classes = 28

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# load the data
x_train, y_train = np.load("../processed_data/splitted_data/x_train.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_train.npy", allow_pickle=True)
x_val, y_val = np.load("../processed_data/splitted_data/x_val.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_val.npy", allow_pickle=True)
x_test, y_test = np.load("../processed_data/splitted_data/x_test.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_test.npy", allow_pickle=True)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)


#%% ------------------------------ Convert to Tensor, DataLoader generation ----------------------------------
train_data_transform = transforms.Compose([
    transforms.ToTensor(),])
test_data_transform = transforms.Compose([
    transforms.ToTensor(),])

batch_train = 128
batch_test = 512

# apply conversion to tensor
trainset = DataAug(x_train, y_train, transform = train_data_transform ,length=len(x_train))
valset = DataAug(x_val, y_val, transform = test_data_transform, length=len(x_val))
testset = DataAug(x_test, y_test, transform = test_data_transform, length=len(x_test))

# generate dataloader
trainloader = DataLoader(trainset, batch_size=batch_train)
valloader = DataLoader(valset, batch_size=batch_test)
testloader = DataLoader(testset, batch_size=batch_test)

# check sample size
print(len(trainloader.sampler))
print(len(valloader.sampler))
print(len(testloader.sampler))



#%% -------------------------------- Model Architecture ------------------------------------------------------

model = models.resnet34(pretrained=True)

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



#%% --------------------------------- Preparation before training ----------------------------------------
# loss function, epochs, lr, optimizer
criterion_BCELog = nn.BCEWithLogitsLoss()
epochs = 1000
LR = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


#%%------------------------------ Training and fine-tuning model ---------------------------------------------
path ="../Model/model_resnet34_baseline.pt"
model, val_losses, running_f1, running_f1_sample = train_baseline_model(model, criterion_BCELog, optimizer, epochs, "train_val",trainloader, valloader,  path)


#%%------------------------------- Final evaluation on testing set ------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
running_preds = []
running_targets = []
model.eval()  # change to evaluate mode
for data, label in testloader:
    data = data.to(device)
    label = label.to(device)

    true_targets = label.cpu().detach().numpy()
    for true_target in true_targets:
        running_targets.append(true_target)

    with torch.no_grad():
        test_logits = model(data)
        probs = torch.sigmoid(test_logits)
        probs = torch.round(probs)
        probs = probs.cpu().detach().numpy()

    for prob in probs:
        running_preds.append(prob)

running_preds = np.array(running_preds)
running_targets = np.array(running_targets)
F1_macro = f1_score(running_targets, running_preds, average='macro')
F1_sample = f1_score(running_targets, running_preds, average='samples')

print("Macro F1-score on testing set is: {:.6f}".format(F1_macro))
print("Samples F1-score on testing set is: {:.6f}".format(F1_sample))


#%% ---------------------------------------- Learning curve ----------------------------------------------------------
inds = np.arange(1,len(running_f1)+1)
plt.figure()
plt.plot(inds.astype(np.uint8), val_losses, label = "BCEWithLogits loss", linestyle='--', marker='o')
plt.plot(inds.astype(np.uint8), running_f1, label = "Macro F-1 score", linestyle='--', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Magnitude")
plt.title("Baseline resenet34 with learning rate={:.4f}".format(LR))
plt.legend(loc='best')
plt.savefig("../Figure/learning_curve_baseline.pdf")
plt.show()