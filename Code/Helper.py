import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import f1_score

#%%-------------------- Data Augmentation -----------------
class DataAug(Dataset):
    """
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self, data, targets, transform=None, length=None):

        self.transform = transform
        self.data = data.astype(np.uint8)
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.length = length

    def __getitem__(self, index):
        index = index % len(self.data)
        x = self.data[index]
        y = self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        return x , y

    def __len__(self):
        return self.length


#%%------------------ Training processs ---------------------------

def train_baseline_model(model, criterion, optimizer, epochs, mode, trainloader, testloader, path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    count = 0
    train_losses, val_losses = [], []
    running_f1 = []
    running_f1_sample = []


    print("Start training ...")

    for i in range(epochs):
        F1_macro = 0.0
        F1_sample = 0.0
        #print("\n")
        print("=" * 30)
        print('Epoch {}/{}'.format(i+1, epochs))
        print('-' * 30)

        train_loss = 0.0
        val_loss = 0.0


        # set model to training mode
        model.train()

        for data, label in trainloader:
            data = data.to(device)
            label = label.to(device)

            # make gradients zero
            optimizer.zero_grad()
            # predictions
            train_logits = model(data)
            # loss function
            loss = criterion(train_logits, label)
            # backpropagation
            loss.backward()
            # parameter update
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        # track the loss for each epoch
        train_loss = train_loss/len(trainloader.sampler)
        train_losses.append(train_loss)

        if mode == "train":
            if i == 0:
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, path)
                best_loss = train_loss

            else:
                if  train_loss < best_loss:
                    best_loss = train_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, path)
                    count = 0
                else:
                    count = count + 1
            print('Epoch Loss: {:.6f}'.format(train_loss))

        else:
            running_preds = []
            running_targets = []
            # validation model
            model.eval()  # change to evaluate mode
            for data, label in testloader:
                data = data.to(device)
                label = label.to(device)

                true_targets = label.cpu().detach().numpy()
                for true_target in true_targets:
                    running_targets.append(true_target)

                with torch.no_grad():
                    val_logits = model(data)
                    probs = torch.sigmoid(val_logits)
                    probs = torch.round(probs)
                    probs = probs.cpu().detach().numpy()
                    loss = criterion(val_logits, label)
                    val_loss += loss.item()*data.size(0)

                for prob in probs:
                    running_preds.append(prob)

            running_preds = np.array(running_preds)
            running_targets = np.array(running_targets)
            F1_macro = f1_score(running_targets, running_preds, average='macro')
            F1_sample = f1_score(running_targets, running_preds, average='samples')
            running_f1.append(F1_macro)
            running_f1_sample.append(F1_sample)

            # track the loss
            val_loss = val_loss / len(testloader.sampler)
            val_losses.append(val_loss)


            # save model
            if i == 0:
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, path)
                best_loss = val_loss
                best_macro_F1 = F1_macro
                best_sample_F1 = F1_sample

            else:
                if  val_loss < best_loss:
                    best_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_macro_F1 = F1_macro
                    best_sample_F1 = F1_sample
                    torch.save(best_model_wts, path)
                    count = 0
                else:
                    count = count + 1
            print('Epoch Loss: {:.6f} |  Validation BCE Loss: {:.6f} | Macro F1: {:.6f} | Sample F1: {:.6f}'.format(
                train_loss, val_loss, F1_macro, F1_sample))

        if mode == "train":
            if (count == 10) or (best_loss < 1e-3):
                break
        else:
            if count == 5:
                break


    print("=" * 20)
    print("Training Complete.")
    print("Best_loss: {:.6f} | Macro F1-score: {:.6f} | Sample F1-score: {:.6f}".format(best_loss, best_macro_F1, best_sample_F1))

    # load best model weights
    model.load_state_dict(torch.load(path))
    return model, val_losses, running_f1, running_f1_sample

# ========================
def train_model(model, criterion1, criterion2, optimizer, epochs, mode, trainloader, testloader, path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion1 = criterion1.to(device)
    criterion2 = criterion2.to(device)
    count = 0
    train_losses, val_losses, val_losses_ex = [], [], []
    running_f1 = []
    running_f1_sample = []


    print("Start training ...")

    for i in range(epochs):
        #print("\n")
        F1_macro = 0.0
        F1_sample = 0.0
        print("=" * 30)
        print('Epoch {}/{}'.format(i+1, epochs))
        print('-' * 30)

        train_loss = 0.0
        val_loss = 0.0
        val_loss_ex = 0.0

        # set model to training mode
        model.train()

        for data, label in trainloader:
            data = data.to(device)
            label = label.to(device)

            # make gradients zero
            optimizer.zero_grad()
            # predictions
            train_logits = model(data)
            # loss function
            loss = criterion1(train_logits, label)
            # backpropagation
            loss.backward()
            # parameter update
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        # track the loss for each epoch
        train_loss = train_loss/len(trainloader.sampler)
        train_losses.append(train_loss)

        if mode == "train":
            if i == 0:
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, path)
                best_loss = train_loss

            else:
                if  train_loss < best_loss:
                    best_loss = train_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, path)
                    count = 0
                else:
                    count = count + 1
            print('Epoch Loss: {:.6f}'.format(train_loss))

        else:
            running_preds = []
            running_targets = []
            # validation model
            model.eval()  # change to evaluate mode
            for data, label in testloader:
                data = data.to(device)
                label = label.to(device)

                true_targets = label.cpu().detach().numpy()
                for true_target in true_targets:
                    running_targets.append(true_target)

                with torch.no_grad():
                    val_logits = model(data)
                    probs = torch.sigmoid(val_logits)
                    probs = torch.round(probs)
                    probs = probs.cpu().detach().numpy()
                    loss = criterion2(val_logits, label)
                    loss_ex = criterion1(val_logits, label)
                    val_loss += loss.item()*data.size(0)
                    val_loss_ex += loss_ex.item()*data.size(0)

                for prob in probs:
                    running_preds.append(prob)

            running_preds = np.array(running_preds)
            running_targets = np.array(running_targets)
            F1_macro = f1_score(running_targets, running_preds, average='macro')
            F1_sample = f1_score(running_targets, running_preds, average='samples')
            running_f1.append(F1_macro)
            running_f1_sample.append(F1_sample)
            # track the loss
            val_loss = val_loss / len(testloader.sampler)
            val_losses.append(val_loss)

            val_loss_ex = val_loss_ex / len(testloader.sampler)
            val_losses_ex.append(val_loss_ex)


            # save model
            if i == 0:
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, path)
                best_loss = val_loss
                best_macro_F1 = F1_macro
                best_sample_F1 = F1_sample

            else:
                if  val_loss < best_loss:
                    best_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_macro_F1 = F1_macro
                    best_sample_F1 = F1_sample
                    torch.save(best_model_wts, path)
                    count = 0
                else:
                    count = count + 1
            print('Epoch Loss: {:.6f} |  Validation BCE Loss: {:.6f}  |  Validation Focal Loss: {:.6f} | Macro F1-score: {:.6f} | Sample F1-score: {:.6f}'.format(
                train_loss, val_loss, val_loss_ex, F1_macro, F1_sample))

        if mode == "train":
            if (count == 10) or (best_loss < 1e-3):
                break
        else:
            if count == 5:
                break


    print("=" * 20)
    print("Training Complete.")
    print("Best_loss: {:.6f} | Macro F1-score: {:.6f} | Sample F1-score: {:.6f}".format(best_loss, best_macro_F1, best_sample_F1))

    # load best model weights
    model.load_state_dict(torch.load(path))
    return model, val_losses, val_losses_ex, running_f1, running_f1_sample

#%%------------------------ Focal Loss -----------------------------
class FocalLoss(nn.Module):
    """
    base source code:
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    """

    def __init__(self, alpha=0.25, gamma=1.5, logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma
        self.logits = logits


    def forward(self, inputs, targets):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.alpha = self.alpha.to(device)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))

        pt = torch.exp(-BCE_loss)
        F_loss = at* (1-pt)**self.gamma * BCE_loss

        return torch.mean(F_loss)

