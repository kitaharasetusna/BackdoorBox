# 1. train BadNet on tinyImageNet 

import sys
import torch.nn.functional as F
import copy
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models

sys.path.append('..')
import core
from myutils import utils_data, utils_attack, utils_defence, tiny_imagenet_dataset


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}')
    return val_loss/len(val_loader), 100 * correct / total

# ----------------------------------------- 1 fix seed -----------------------------------------
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 2 configs  ------------------------------------------
exp_dir = '../experiments/exp7_TinyImageNet/Badnet' 
secret_size = 20; label_backdoor = 6; triggerX = 6; triggerY=6 
bs_tr = 256; epoch_Badnet = 300; lr_Badnet = 1e-4
os.makedirs(exp_dir, exist_ok=True)
train_detecor = True

# ----------------------------------------- 3 load model ------------------------------------------
device = torch.device("cuda:0")
# Load a pretrained ResNet-18 model
model = models.resnet18(pretrained=True)
model = torchvision.models.get_model('resnet18', num_classes=200)
model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model.maxpool = nn.Identity()
model = model.to(device)
if not train_detecor:
    model.load_state_dict(torch.load(exp_dir+'/checkpoint.pth')['model'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr_Badnet)


# ----------------------------------------- 4 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_ImageNet_datasets(foloder=exp_dir,
                                load=True)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ids_q)}"

# ----------------------------------------- 5 train model 
ds_questioned = utils_attack.CustomCIFAR10Badnet(
    ds_tr, ids_q, ids_p, label_backdoor, triggerY=triggerY, triggerX=triggerX)
dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
    num_workers=0,drop_last=False,
)
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)

ACC_, ASR_ = utils_attack.test_asr_acc_badnet(dl_te=dl_te, model=model,
                label_backdoor=label_backdoor, triggerX=triggerX, triggerY=triggerY,
                device=device) 

model.train()
ACC = []; ASR= []

for epoch_ in range(epoch_Badnet):
    for inputs, targets in dl_x_q:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # make a forward pass
        outputs = model(inputs)
        # calculate the loss
        loss = criterion(outputs, targets)
        # do a backwards pass
        loss.backward()
        # perform a single optimization step
        optimizer.step()
    print(f'epoch: {epoch_+1}')
    if (epoch_+1)%5==0 or epoch_==0 or epoch_==epoch_Badnet-1:
        model.eval()
        ACC_, ASR_ = utils_attack.test_asr_acc_badnet(dl_te=dl_te, model=model,
                        label_backdoor=label_backdoor, triggerX=triggerX, triggerY=triggerY,
                        device=device) 
        ACC.append(ACC_); ASR.append(ASR_)
        if not train_detecor:
            torch.save(model.state_dict(), exp_dir+'/'+f'step1_model_{epoch_+1}.pth')
        else:
            torch.save(model.state_dict(), exp_dir+'/'+f'step1_model_detector.pth')
            sys.exit(0)
        with open(exp_dir+f'/step1_train_badnet.pkl', 'wb') as f:
            pickle.dump({'ACC': ACC, 'ASR': ASR },f)
        model.train()


    
