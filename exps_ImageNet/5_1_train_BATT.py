# 1. train BATT on CIFAR-10


# step1: train model

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import os
import pickle
import lpips
import numpy as np
sys.path.append('..')
import core
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator
from myutils import utils_data, utils_attack, utils_defence
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
import torchvision


# ----------------------------------------- 1 fix seed -----------------------------------------
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 2 configs  ------------------------------------------
exp_dir = '../experiments/exp7_TinyImageNet/BATT' 
dataset = 'tiny_img'
label_backdoor = 6
bs_tr = 128; epoch_BATT = 100; lr_BATT = 1e-4
rotation = 16; train_detecor = False 
# ----------------------------------------- 0.2 dirs, load ISSBA_encoder+secret
# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)
device = torch.device("cuda:0")

model = models.resnet18(pretrained=True)
model = torchvision.models.get_model('resnet18', num_classes=200)
model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model.maxpool = nn.Identity()
model = model.to(device)
if not train_detecor:
    model.load_state_dict(torch.load(exp_dir+'/checkpoint.pth')['model'])
optimizer = torch.optim.Adam(model.parameters(), lr=lr_BATT)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.1)
criterion = nn.CrossEntropyLoss()

normalization = utils_defence.get_dataset_normalization(dataset)
denormalization = utils_defence.get_dataset_denormalization(normalization)
# ----------------------------------------- 0.3 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_ImageNet_datasets_batt(foloder=exp_dir,
                                load=False)
# ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_CIFAR10_datasets_batt(foloder=exp_dir,
#                                 load=True)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ids_q)}"

# ----------------------------------------- train model with ISSBA encoder
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)
ds_questioned = utils_attack.CustomCIFAR10BATT(original_dataset=ds_tr, subset_indices=ids_q,
                                               trigger_indices=ids_p, label_bd=label_backdoor, roation=rotation)
dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
    num_workers=0,drop_last=False,
)

model.train()
ACC = []; ASR= []
for epoch_ in range(epoch_BATT):
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
    scheduler2.step()
    print(f'epoch: {epoch_+1}')
    # if (epoch_+1)%5==0 or epoch_==0 or epoch_==epoch_Blended-1:
    if True:
        model.eval()
        # TODO: change this to BATT 
        ACC_, ASR_ = utils_attack.test_asr_acc_batt(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                    rotation=rotation, device=device)
        ACC.append(ACC_); ASR.append(ASR_)
        torch.save(model.state_dict(), exp_dir+'/'+f'step1_model_{epoch_+1}.pth')
        with open(exp_dir+f'/step1_train_BATT.pkl', 'wb') as f:
            pickle.dump({'ACC': ACC, 'ASR': ASR },f)
        model.train()