# ======================= RESNET 34 with Focal Loss, Oversampling and Augmentation ==============================
#%% -------------------------------- Import Lib ----------------------------------------------------------------------
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
from sklearn.utils import shuffle


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
x_train_over, y_train_over = np.load("../processed_data/splitted_data/x_train_over.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_train_over.npy", allow_pickle=True)
x_val, y_val = np.load("../processed_data/splitted_data/x_val.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_val.npy", allow_pickle=True)
x_test, y_test = np.load("../processed_data/splitted_data/x_test.npy", allow_pickle=True), np.load("../processed_data/splitted_data/y_test.npy", allow_pickle=True)

# shuffling
x_train, y_train = shuffle(x_train, y_train)
x_train_over, y_train_over = shuffle(x_train_over, y_train_over)

# shape view
print(x_train_over.shape, y_train_over.shape)
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)


#%% --------------------------------- Data Augmentation, Dataloader ------------------------------------------------------------------
# define transformations
## training set
train_data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(120),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=90),
    transforms.ToTensor(),
])

## validation and testing sets
test_data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(120),
    transforms.ToTensor(),

])

# batch size
batch_train = 512
batch_test = 512

# apply transformation
trainset_over = DataAug(x_train_over, y_train_over, transform = train_data_transform ,length=3*len(x_train_over))
trainset = DataAug(x_train, y_train, transform = test_data_transform ,length=len(x_train))

# combining original training set and oversampled set
trainset = torch.utils.data.ConcatDataset([trainset_over,trainset])

# validation and testing set
valset = DataAug(x_val, y_val, transform = test_data_transform, length=len(x_val))
testset = DataAug(x_test, y_test, transform = test_data_transform, length=len(x_test))

# generating dataloader
trainloader = DataLoader(trainset, batch_size=batch_train)
valloader = DataLoader(valset, batch_size=batch_test)
testloader = DataLoader(testset, batch_size=batch_test)

# check sample size
print(len(trainloader.sampler))
print(len(valloader.sampler))
print(len(testloader.sampler))



#%% --------------------------------- Model Architecture -------------------------------------------------

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
    nn.Dropout(0.5),
    nn.Linear(256, n_classes)
    )
for param in model.parameters():
    param.requires_grad = True


#%% ------------------------------------ Preparation ---------------------------------------------------
# loss function, lr and epochs
criterion_Focal= FocalLoss()
criterion_BCELog = nn.BCEWithLogitsLoss()
epochs = 1000
LR = 1e-3
# optimizer
optimizer = torch.optim.Adam([
{ 'params': model.conv1.parameters()},
{ 'params': model.bn1.parameters(), 'lr': LR/ 9},
{ 'params': model.maxpool.parameters(), 'lr': LR/ 9},
{ 'params': model.layer1.parameters(), 'lr': LR/ 9},
{ 'params': model.layer2.parameters(), 'lr': LR/ 9},
{ 'params': model.layer3.parameters(), 'lr': LR/ 3},
{ 'params': model.layer4.parameters(), 'lr': LR/ 3},
{ 'params': model.avgpool.parameters(),  'lr': LR/ 3},
{ 'params': model.fc.parameters()} ], lr=LR)

#%%------------------------- Rraining and fine-tuning model ---------------------------------------
path ="../Model/model_resnet34_2.pt"
model, val_losses, val_losses_ex, running_f1, running_f1_sample = train_model(model, criterion_Focal, criterion_BCELog, optimizer, epochs, "train_val",trainloader, valloader,  path)

#%%--------------------------- Final evaluation on testing set -----------------------------------------
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
print("Sample F1-score on testing set is: {:.6f}".format(F1_sample))


#%% ------------------------------------- Learning curve -------------------------------------------
inds = np.arange(1,len(running_f1)+1)
plt.figure()
plt.plot(inds.astype(np.uint8), val_losses, label = "Validation loss")
plt.plot(inds.astype(np.uint8), running_f1, label = "Macro F-1 score")
plt.xlabel("Epoch")
plt.ylabel("Magnitude")
plt.title("Learning curve of resenet34_2 with base learning_rate={:.4f}".format(LR))
plt.legend(loc='best')
plt.savefig("../Figure/learning_curve_resnet34_2.pdf")
plt.show()