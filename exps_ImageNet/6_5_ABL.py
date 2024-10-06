# step3 and step4: train B theta ;break up suprious relationships

import sys
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import os
import pickle
import lpips
import numpy as np
import pickle
import lpips
sys.path.append('..')
import core
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator
from myutils import utils_data, utils_attack, utils_defence
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchvision import models

def comp_inf_norm(A, B):
    infinity_norm = torch.max(torch.abs(A - B))
    return infinity_norm

# Helper function to reset iterator if needed
def get_next_batch(loader_iter, loader):
    try:
        return next(loader_iter)
    except StopIteration:
        return next(iter(loader))

# ----------------------------------------- 0 fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 1 configs:
exp_dir = '../experiments/exp7_TinyImageNet/SIG' 
dataset = 'tiny_img'
label_backdoor = 6
bs_tr = 128
bs_tr2 = 128 # TODO: CHECK THIS 
sig_delta = 40; sig_f = 6
lr_ft = 1e-4; epoch_root = 10
alpha=0.2
normalization = utils_defence.get_dataset_normalization(dataset)
denormalization = utils_defence.get_dataset_denormalization(normalization)
# ----------------------------------------- 0.2 dirs, load ISSBA_encoder+secret+model f'
# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)

device = torch.device("cuda:0")

# Load a pretrained ResNet-18 model
model = models.resnet18(pretrained=True)
model = torchvision.models.get_model('resnet18', num_classes=200)
model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model.maxpool = nn.Identity()
model = model.to(device)
model.load_state_dict(torch.load(exp_dir+f'/step1_model_1.pth'))
criterion = nn.CrossEntropyLoss()

model.eval()
model.requires_grad_(False)


# ----------------------------------------- 0.3 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_ImageNet_datasets_SIG(foloder=exp_dir,
                                load=True)
# ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_CIFAR10_datasets_batt(foloder=exp_dir,
#                                 load=True)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ids_q)}"

# ----------------------------------------- train model with ISSBA encoder
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)
ACC_, ASR_ = utils_attack.test_asr_acc_sig(dl_te=dl_te, model=model,
                                                   label_backdoor=label_backdoor,
                                                   delta=sig_delta, freq=sig_f, device=device,
                                                   norm=normalization, denorm=denormalization)

with open(exp_dir+'/idx_suspicious.pkl', 'rb') as f:
    idx_sus = pickle.load(f)

print('number of suspicious samples: ', len(idx_sus))
TP, FP = 0.0, 0.0
for s in idx_sus:
    if s in ids_p:
        TP+=1
    else:
        FP+=1
print(TP/(TP+FP))

# ----------------------------------------- 1 train B_theta  
# prepare B
ds_whole_poisoned = utils_attack.CustomCIFAR10SIG_whole(ds_tr, ids_p, label_backdoor, sig_delta, sig_f,
                                                        norm=normalization, denorm=denormalization)

# B_theta = utils_attack.FixedSTN(input_channels=3, device=device)
ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_tr2,shuffle=True,num_workers=0,drop_last=True)
# TODO: change this
ds_sus = Subset(ds_whole_poisoned, idx_sus)
dl_sus = DataLoader(dataset= ds_sus,batch_size=bs_tr2,shuffle=True,num_workers=0,drop_last=True)

loader_root_iter = iter(dl_root); loader_sus_iter = iter(dl_sus) 
optimizer = torch.optim.Adam(model.parameters(), lr=lr_ft)
print(len(ds_sus), len(ds_x_root))


model.train(); model.requires_grad_(True)
ACC = []; ASR= []
for epoch_ in range(epoch_root):
    for i in range(max(len(dl_root), len(dl_sus))):
        X_root, Y_root = get_next_batch(loader_root_iter, dl_root)
        X_q, Y_q = get_next_batch(loader_sus_iter, dl_sus)
        X_root, Y_root = X_root.to(device), Y_root.to(device)
        X_q, Y_q= X_q.to(device), Y_q.to(device)
        optimizer.zero_grad()
        # make a forward pass
        Y_root_pred = model(X_root)
        Y_q_pred = model(X_q)
        # calculate the loss
        loss = criterion(Y_root_pred, Y_root)-criterion(Y_q_pred, Y_q)
        loss = criterion(Y_root_pred, Y_root)
        # do a backwards pass
        loss.backward()
        # perform a single optimization step
        optimizer.step()
    print(f'epoch: {epoch_+1}')
    if True:
        model.eval()
        ACC_, ASR_ = utils_attack.test_asr_acc_sig(dl_te=dl_te, model=model,
                                                   label_backdoor=label_backdoor,
                                                   delta=sig_delta, freq=sig_f, device=device,
                                                   norm=normalization, denorm=denormalization) 
        ACC.append(ACC_); ASR.append(ASR_)
        torch.save(model.state_dict(), exp_dir+'/'+f'model_sus_cln_{epoch_+1}.pth')
        with open(exp_dir+f'/root_model_clean.pkl', 'wb') as f:
            pickle.dump({'ACC': ACC, 'ASR': ASR },f)
        model.train()


