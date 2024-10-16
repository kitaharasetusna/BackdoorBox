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


def get_train_fim_ISSBA(model, dl_train, encoder, secret, ratio_poison, bs_tr, device):
    ''' get FIM while training on training data
        returns:
            avg_trace_fim, avg_trace_fim_bd, avg_loss, avg_loss_bd 
            
    '''
    num_poison = int(ratio_poison*bs_tr)
    avg_trace_fim = 0.0; avg_trace_fim_bd = 0.0; avg_loss = 0.0; avg_loss_bd = 0.0
    cln_num = 0; bd_num = 0
    for images, labels in dl_train:
        cln_num+=(bs_tr-num_poison); bd_num+=num_poison
        trace_fim_cln, loss_cln = utils_defence.compute_fisher_information(model, images[num_poison:], 
                                                                labels[num_poison:], criterion,
                                                                device= device, loss_=True)
        avg_trace_fim += trace_fim_cln; avg_loss+=loss_cln
        inputs_bd, targets_bd = copy.deepcopy(images), copy.deepcopy(labels)
        for xx in range(num_poison):
            inputs_bd[xx] = utils_attack.add_ISSBA_trigger(inputs=inputs_bd[xx], secret=secret,
                                                           encoder=encoder, device=device)
            # inputs_bd[xx] = utils_attack.add_badnet_trigger(inputs=inputs_bd[xx], triggerY=triggerY,
            #                                                 triggerX=triggerX) 
            targets_bd[xx] = label_backdoor
        trace_fim_bd, loss_bd = utils_defence.compute_fisher_information(model, inputs_bd[:num_poison], 
                                                                    targets_bd[:num_poison], criterion,
                                                                    device=device, loss_=True)
        avg_trace_fim_bd += trace_fim_bd; avg_loss_bd+=loss_bd
    avg_trace_fim = avg_trace_fim/(1.0*cln_num); avg_trace_fim_bd = avg_trace_fim_bd/(1.0*bd_num)
    avg_loss = avg_loss/(1.0*cln_num); avg_loss_bd = avg_loss_bd/(1.0*bd_num)
    print(f'fim clean: {avg_trace_fim: .2f}')
    print(f'fim bd: {avg_trace_fim_bd: .2f}')   
    print(f'loss clean: {avg_loss: 2f}')
    print(f'loss bd: {avg_loss_bd: .2f}')
    return avg_trace_fim, avg_trace_fim_bd, avg_loss, avg_loss_bd

# ----------------------------------------- 0.0 fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 0.1 configs:
exp_dir = '../experiments/exp7_TinyImageNet/Blended' 
secret_size = 20; label_backdoor = 6
bs_tr = 128
idx_blend = 656
bs_tr2 = 50 
lr_B = 1e-4;epoch_B = 100 
lr_ft = 1e-4
alpha=0.2
lr_root = 1e-4; epoch_root = 10
# ----------------------------------------- 0.2 dirs, load ISSBA_encoder+secret+model f'
# make a directory for experimental results
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
# ----------------------------------------- 0.3 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_ImageNet_datasets(foloder=exp_dir,
                                load=False)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ids_q)}"

pattern, _ = ds_te[idx_blend] #(3, 32, 32)
ds_questioned = utils_attack.CustomCIFAR10Blended(original_dataset=ds_tr, subset_indices=ids_q,
                trigger_indices=ids_p, label_bd=label_backdoor, pattern=pattern, alpha=alpha)

dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
    num_workers=0,drop_last=False,
)
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)

ACC_, ASR_ = utils_attack.test_asr_acc_blended(dl_te=dl_te, model=model,
                        label_backdoor=label_backdoor, pattern=pattern, device=device, alpha=alpha) 
print(ACC_, ASR_)
with open(exp_dir+'/idx_suspicious.pkl', 'rb') as f:
    idx_sus = pickle.load(f)
TP, FP = 0.0, 0.0
for s in idx_sus:
    if s in ids_p:
        TP+=1
    else:
        FP+=1
print(TP/(TP+FP))
# ----------------------------------------- 1 train B_theta  

# prepare B
ds_whole_poisoned = utils_attack.CustomCIFAR10Blended_whole(ds_tr, ids_p, label_bd=label_backdoor, pattern=pattern, alpha=alpha)
ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_tr,shuffle=True,num_workers=0,drop_last=False)
# TODO: change this
ds_sus = Subset(ds_whole_poisoned, idx_sus)
dl_sus = DataLoader(dataset= ds_sus,batch_size=bs_tr,shuffle=True,num_workers=0,drop_last=False)

loader_root_iter = iter(dl_root); loader_sus_iter = iter(dl_sus) 

# -------------------------------------------- train backdoor using B theta on a clean model
model_root = core.models.ResNet(18); model_root = model_root.to(device)
criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=lr_root)

loader_root_iter = iter(dl_root); loader_sus_iter = iter(dl_sus) 

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
        # do a backwards pass
        loss.backward()
        # perform a single optimization step
        optimizer.step()
    print(f'epoch: {epoch_+1}')
    if True:
        model.eval()
        ACC_, ASR_ = utils_attack.test_asr_acc_blended(dl_te=dl_te, model=model,
                        label_backdoor=label_backdoor, pattern=pattern, device=device, alpha=alpha)
        ACC.append(ACC_); ASR.append(ASR_)
        torch.save(model.state_dict(), exp_dir+'/'+f'model_sus_cln_{epoch_+1}.pth')
        with open(exp_dir+f'/root_model_clean.pkl', 'wb') as f:
            pickle.dump({'ACC': ACC, 'ASR': ASR },f)
        model.train()
