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
from torch.utils.data import Subset
sys.path.append('..')
import core
from core.attacks.IAD import Generator
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator
from myutils import utils_data, utils_attack, utils_defence
import matplotlib.pyplot as plt
import random

def get_dataloader_order(dataloader):
    order = []
    for batch in dataloader:
        inputs, targets = batch
        order.extend(targets.tolist())
    return order

# ----------------------------------------- 0.0 fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 0.1 configs:
exp_dir = '../experiments/exp6_FI_B/IAD' 
label_backdoor = 6
bs_tr = 128; epoch_IAD = 20; lr_IAD = 1e-4
epoch_M = 10; episilon=1e-7; lambda_div=1; mask_density=0.032; lambda_norm=100
# ----------------------------------------- 0.2 dirs, load ISSBA_encoder+secret
# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)

device = torch.device("cuda:0")

model = core.models.ResNet(18); model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_IAD)
criterion = nn.CrossEntropyLoss()

# ----------------------------------------- 0.3 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_CIFAR10_datasets_2(foloder=exp_dir,
                                load=True)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ds_q)}"


#--------------------------------------------train M and G and model 
modelM = Generator('cifar10', out_channels=1).to(device)
optimizerM = torch.optim.Adam(modelM.parameters(), lr=0.01, betas=(0.5, 0.9))
schedulerM = torch.optim.lr_scheduler.MultiStepLR(optimizerM, [10, 20], 0.1)

modelG = Generator('cifar10').to(device)
optimizerG = torch.optim.Adam(modelG.parameters(), lr=0.01, betas=(0.5, 0.9))
schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, [200, 300, 400, 500], 0.1)

# train M first
load_idx_M = True
if load_idx_M == False:
    sample_size = int(len(ids_q) * 0.1)
    ids_M = random.sample(ids_q, sample_size)
    with open(exp_dir+'/step1_M_idx.pkl', 'wb') as f:
        pickle.dump(ids_M, f)
else:
    with open(exp_dir+'/step1_M_idx.pkl', 'rb') as f:
        ids_M = pickle.load(f)

ds_M = Subset(ds_tr, ids_M) 

dl_M_1 = DataLoader(dataset=ds_M,batch_size=bs_tr,shuffle=True,num_workers=0,drop_last=True)
dl_M_2= DataLoader(dataset=ds_M,batch_size=bs_tr,shuffle=True,num_workers=0,drop_last=True)

order_1 = get_dataloader_order(dl_M_1); order_2 = get_dataloader_order(dl_M_2); 
assert order_1!=order_2, "ds_tr for training diversity loss failed to have different order" 
for epoch_m_ in range(epoch_M):
    modelM.train()
    total_loss = 0
    criterion_div = nn.MSELoss(reduction="none")
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(dl_M_1)), dl_M_1, dl_M_2):
        optimizerM.zero_grad()

        inputs1, targets1 = inputs1.to(device), targets1.to(device)
        inputs2, targets2 = inputs2.to(device), targets2.to(device)

        # Generate the mask of data
        masks1, masks2 = modelM.threshold(modelM(inputs1)), modelM.threshold(modelM(inputs2))

        # Calculating diversity loss
        distance_images = criterion_div(inputs1, inputs2)
        distance_images = torch.mean(distance_images, dim=(1, 2, 3))
        distance_images = torch.sqrt(distance_images)

        distance_patterns = criterion_div(masks1, masks2)
        distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
        distance_patterns = torch.sqrt(distance_patterns)

        loss_div = distance_images / (distance_patterns + episilon)
        loss_div = torch.mean(loss_div) * lambda_div

        # Calculating mask magnitude loss
        loss_norm = torch.mean(F.relu(masks1 - mask_density))

        total_loss = lambda_norm * loss_norm + lambda_div * loss_div
        total_loss.backward()
        optimizerM.step()
    schedulerM.step()
    msg = f"epoch: {epoch_m_+1} " + "Train Mask loss: {:.4f} | Norm: {:.3f} | Diversity: {:.3f}\n".format(total_loss, loss_norm, loss_div)
    print(msg)
torch.save(modelM.state_dict(), exp_dir+'/step1_M.pth')
modelM.eval()
modelM.requires_grad_(False)
# TODO: train G and model
sys.exit(0)



# -----------------------------------------collect dataset 
# TODO: change dataset below
ds_questioned = utils_attack.CustomCIFAR10Badnet(
    ds_tr, ids_q, ids_p, label_backdoor, triggerY=triggerY, triggerX=triggerX)
dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
    num_workers=0,drop_last=False,
)
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)

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
    # TODO: make this get_train_fm_ISSBA 
    print(f'epoch: {epoch_+1}')
    if (epoch_+1)%5==0 or epoch_==0 or epoch_==epoch_Badnet-1:
        model.eval()
        ACC_, ASR_ = utils_attack.test_asr_acc_badnet(dl_te=dl_te, model=model,
                        label_backdoor=label_backdoor, triggerX=triggerX, triggerY=triggerY,
                        device=device) 
        ACC.append(ACC_); ASR.append(ASR_)
        torch.save(model.state_dict(), exp_dir+'/'+f'step1_model_{epoch_+1}.pth')
        with open(exp_dir+f'/step1_train_badnet.pkl', 'wb') as f:
            pickle.dump({'ACC': ACC, 'ASR': ASR },f)
        model.train()
