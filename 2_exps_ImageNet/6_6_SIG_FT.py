# verify that B_theta learn the same as malicious samples
import sys
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import os
import pickle
import lpips
import numpy as np
import pickle
sys.path.append('..')
import core
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator
from myutils import utils_data, utils_attack, utils_defence
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import lpips
from torchvision import models

# Helper function to reset iterator if needed
def get_next_batch(loader_iter, loader):
    try:
        return next(loader_iter)
    except StopIteration:
        return next(iter(loader))
# ----------------------------------------- 0.0 fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 0.3 prepare data X_root X_questioned
exp_dir = '../experiments/exp7_TinyImageNet/SIG' 
dataset = 'tiny_img'
label_backdoor = 6
bs_tr = 128; epoch_SIG = 100; lr_SIG = 1e-3
bs_tr2 = 128
sig_delta = 40; sig_f = 6
lr_root = 1e-4; epoch_root = 10
normalization = utils_defence.get_dataset_normalization(dataset)
denormalization = utils_defence.get_dataset_denormalization(normalization)
# ----------------------------------------- 1 train B_theta  
os.makedirs(exp_dir, exist_ok=True)

device = torch.device("cuda:0")

model = core.models.ResNet(18); model = model.to(device)

model = models.resnet18(pretrained=True)
model = torchvision.models.get_model('resnet18', num_classes=200)
model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model.maxpool = nn.Identity()
model = model.to(device)
model.load_state_dict(torch.load(exp_dir+'/step1_model_1.pth'))
criterion = nn.CrossEntropyLoss()

model.eval()
model.requires_grad_(False)


# ----------------------------------------- 0.3 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_ImageNet_datasets_SIG(foloder=exp_dir,
                                load=True)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ids_q)}"

# ----------------------------------------- train model with ISSBA encoder
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)
ds_questioned = utils_attack.CustomCIFAR10SIG(original_dataset=ds_tr, subset_indices=ids_q+ids_root,
                                               trigger_indices=ids_p, label_bd=label_backdoor,
                                               delta=sig_delta, frequency=sig_f,
                                               norm=normalization, denorm=denormalization)
dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
    num_workers=0,drop_last=False,
)
ACC_, ASR_ = utils_attack.test_asr_acc_sig(dl_te=dl_te, model=model,
                                                   label_backdoor=label_backdoor,
                                                   delta=sig_delta, freq=sig_f, device=device,
                                                   norm=normalization, denorm=denormalization)

# if train_B:
#     with open(exp_dir+'/idx_suspicious.pkl', 'rb') as f:
#         idx_sus = pickle.load(f)
# else:
with open(exp_dir+'/idx_suspicious.pkl', 'rb') as f:
    idx_sus = pickle.load(f)
print(len(idx_sus))
TP, FP = 0.0, 0.0
for s in idx_sus:
    if s in ids_p:
        TP+=1
    else:
        FP+=1
print(TP/(TP+FP))
# ----------------------------------------- 1 train B_theta  

# prepare B
ds_whole_poisoned = utils_attack.CustomCIFAR10SIG_whole(ds_tr, ids_p, label_backdoor, sig_delta, sig_f)
ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_tr2,shuffle=True,num_workers=0,drop_last=True)
# TODO: change this
ds_sus = Subset(ds_whole_poisoned, idx_sus)
dl_sus = DataLoader(dataset= ds_sus,batch_size=bs_tr2,shuffle=True,num_workers=0,drop_last=True)

loader_root_iter = iter(dl_root); loader_sus_iter = iter(dl_sus) 
# -------------------------------------------- train backdoor using B theta on a clean model
model_root = core.models.ResNet(18); model_root = model_root.to(device)
criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=lr_root)

loader_root_iter = iter(dl_root); loader_sus_iter = iter(dl_sus) 

model.train(); model.requires_grad_(True)
ACC = []; ASR= []
for epoch_ in range(epoch_root):
    for X_root, Y_root in dl_root:
        X_root, Y_root = X_root.to(device), Y_root.to(device)
        optimizer.zero_grad()
        # make a forward pass
        Y_root_pred = model(X_root)
        # calculate the loss
        loss = criterion(Y_root_pred, Y_root)
        # do a backwards pass
        loss.backward()
        # perform a single optimization step
        optimizer.step() 
    print(f'epoch: {epoch_+1}')
    if True:
        model.eval()
        ACC_, ASR_ = utils_attack.test_asr_acc_sig(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                freq=sig_f, delta=sig_delta, device=device,
                                                norm=normalization, denorm=denormalization) 
        ACC.append(ACC_); ASR.append(ASR_)
        torch.save(model.state_dict(), exp_dir+'/'+f'model_sus_cln_{epoch_+1}.pth')
        with open(exp_dir+f'/root_model_clean.pkl', 'wb') as f:
            pickle.dump({'ACC': ACC, 'ASR': ASR },f)
        model.train()
