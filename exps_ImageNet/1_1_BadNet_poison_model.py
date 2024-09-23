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

# ----------------------------------------- 3 load model ------------------------------------------
device = torch.device("cuda:0")
# Load a pretrained ResNet-18 model
# Load a pretrained ResNet-18 model
model = models.resnet18(pretrained=True)
model = torchvision.models.get_model('resnet18', num_classes=200)
model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model.maxpool = nn.Identity()
model = model.to(device)
model.load_state_dict(torch.load(exp_dir+'/checkpoint.pth')['model'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ----------------------------------------- 4 prepare data X_root X_questioned
# ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_ImageNet_datasets(foloder=exp_dir,
#                                 load=False)
# print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
# assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
# assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ids_q)}"
# num_classes = len(ds_tr.class_to_idx)
normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
print("Loading training data")
train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
ds_tr = tiny_imagenet_dataset.TinyImageNet('../data', split='train', download=True, transform=train_transform)
print("Loading validation data")
val_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
ds_te = tiny_imagenet_dataset.TinyImageNet('../data', split='val', download=False, transform=val_transform)


# ----------------------------------------- 5 train model 
# ds_questioned = utils_attack.CustomCIFAR10Badnet(
#     ds_tr, ids_q, ids_p, label_backdoor, triggerY=triggerY, triggerX=triggerX)
# dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
#     num_workers=0,drop_last=False,
# )
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)
dl_tr = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=True,
    num_workers=0, drop_last=False
)

validate(model, dl_te, criterion)
# model.eval()
# validate(model, dl_te, criterion)
# validate(model, dl_tr, criterion)
# ACC_, ASR_ = utils_attack.test_asr_acc_badnet(dl_te=dl_tr, model=model,
#                 label_backdoor=label_backdoor, triggerX=triggerX, triggerY=triggerY,
#                 device=device) 
# sys.exit(0)
# model.train()
# ACC = []; ASR= []

# for epoch_ in range(epoch_Badnet):
#     for inputs, targets in dl_x_q:
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         # make a forward pass
#         outputs = model(inputs)
#         # calculate the loss
#         loss = criterion(outputs, targets)
#         # do a backwards pass
#         loss.backward()
#         # perform a single optimization step
#         optimizer.step()
#     # TODO: make this get_train_fm_ISSBA 
#     print(f'epoch: {epoch_+1}')
#     if (epoch_+1)%5==0 or epoch_==0 or epoch_==epoch_Badnet-1:
#         model.eval()
#         ACC_, ASR_ = utils_attack.test_asr_acc_badnet(dl_te=dl_te, model=model,
#                         label_backdoor=label_backdoor, triggerX=triggerX, triggerY=triggerY,
#                         device=device) 
#         ACC.append(ACC_); ASR.append(ASR_)
#         torch.save(model.state_dict(), exp_dir+'/'+f'step1_model_{epoch_+1}.pth')
#         with open(exp_dir+f'/step1_train_badnet.pkl', 'wb') as f:
#             pickle.dump({'ACC': ACC, 'ASR': ASR },f)
#         model.train()

# ----------------------------------------- train model with ISSBA encoder
# ds_questioned = utils_attack.CustomCIFAR10Badnet(
#     ds_tr, ids_q, ids_p, label_backdoor, triggerY=triggerY, triggerX=triggerX)
# dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
#     num_workers=0,drop_last=False,
# )
# dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
#     num_workers=0, drop_last=False
# )

# model.train()
# ACC = []; ASR= []



# # Create dataloaders
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# print(len(train_dataset), len(val_dataset))
# print(len(train_loader), len(val_loader))

# # -----
# # Load pre-trained ResNet-18 model and modify it for 200 classes
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 200)  # Modify the output layer

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)


# # -----
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# Training function
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 100 == 0:
            print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')

    print(f'Train Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct / total}')

# Validation function


# Main training loop
# num_epochs = 1000
# for epoch in range(num_epochs):
#     train(model, dl_x_q, criterion, optimizer, epoch)
#     if epoch%5==0:
#         validate(model, dl_te, criterion)
    
