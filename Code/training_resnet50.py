import torch
import torch.nn as nn
import os
import random
import numpy as np
from Helper import train_model, DataAug, FocalLoss
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

n_classes = 28

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# load the data
# split data into the training set and testing set
x_train, y_train = np.load("../processed_data/splitted_data/x_train.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_train.npy", allow_pickle=True)
x_val, y_val = np.load("../processed_data/splitted_data/x_val.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_val.npy", allow_pickle=True)
x_test, y_test = np.load("../processed_data/splitted_data/x_test.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_test.npy", allow_pickle=True)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)


#%% ---------- Convert to torch.Tensor -----------
train_data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=90),
    transforms.ToTensor(),
])
test_data_transform = transforms.Compose([
    transforms.ToTensor(),

])

batch_train = 128
batch_test = 512

trainset = DataAug(x_train, y_train, transform = train_data_transform ,length=5*len(x_train))
valset = DataAug(x_val, y_val, transform = test_data_transform, length=len(x_val))
testset = DataAug(x_test, y_test, transform = test_data_transform, length=len(x_test))

trainloader = DataLoader(trainset, batch_size=batch_train)
valloader = DataLoader(valset, batch_size=batch_test)
testloader = DataLoader(testset, batch_size=batch_test)

print(len(trainloader.sampler))
print(len(valloader.sampler))
print(len(testloader.sampler))



#%% -------------- Model

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
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Linear(512, n_classes)
    )
for param in model.parameters():
    param.requires_grad = True


#%% --------------
#print(model)
criterion_Focal= FocalLoss()
criterion_Focal.alpha = 0.8
criterion_Focal.gamma = 2.5

criterion_BCELog = nn.BCEWithLogitsLoss()
epochs = 1000
LR = 1e-3
#optimizer = torch.optim.Adam(model.parameters(), lr=LR)
optimizer = torch.optim.Adam([
{ 'params': model.conv1.parameters()},
{ 'params': model.bn1.parameters(), 'lr': LR/ 5},
{ 'params': model.maxpool.parameters(), 'lr': LR/ 5},
{ 'params': model.layer1.parameters(), 'lr': LR/ 5},
{ 'params': model.layer2.parameters(), 'lr': LR/ 5},
{ 'params': model.layer3.parameters(), 'lr': LR/ 3},
{ 'params': model.layer4.parameters(), 'lr': LR/ 3},
{ 'params': model.avgpool.parameters()},
{ 'params': model.fc.parameters()} ], lr=LR)

#%%---------- training and fine-tuning model --------------
path ="../Model/model_resnet34.pt"
model, val_losses, val_losses_ex, running_f1 = train_model(model, criterion_Focal, criterion_BCELog, optimizer, epochs, "train_val",trainloader, valloader,  path)

#%%---------- final evaluation on testing set -------------
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
F1_macro = f1_score(running_targets, running_preds, average='samples')

print("F1-score on testing set is: {:.6f}".format(F1_macro))


#%% ------------ Learning curve --------------------
inds = np.arange(1,len(running_f1)+1)
plt.figure()
plt.plot(inds.astype(np.uint8), val_losses_ex, label = "BCEWithLogits loss", linestyle='--', marker='o')
plt.plot(inds.astype(np.uint8), running_f1, label = "F-1 score", linestyle='--', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Magnitude")
plt.title("Learning curve of resenet50 with base learning_rate={:.4f}".format(LR))
plt.legend(loc='best')
plt.savefig("../Figure/learning_curve_resnet50.pdf")
plt.show()