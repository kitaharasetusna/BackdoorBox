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


# load root data;
# load frozen model
# train B

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

# ----------------------------------------- 0.1 configs:
exp_dir = '../experiments/exp5_G/ISSBA'; label_backdoor = 6
lr_step1 = 1e-4; epoch_step1 = 30; load = False; ratio_poison = 0.1
epoch_encoder = 20; secret_size = 20; enc_secret_only_epoch=2 
train_Encoder = False; verbose = True; sigma = 1e-4

# ----------------------------------------- 0.2 dirs, load poisoned model and ISSBA_encoder+secret
# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)

device = torch.device("cuda:0")
# prepare pretrained model: encoder, resnet-18, secret for ISSBA and freeze them
secret = torch.FloatTensor(np.random.binomial(1, .5, secret_size).tolist()).to(device)
model = core.models.ResNet(18); model = model.to(device)
model.load_state_dict(torch.load(exp_dir+'/'+f'model_ISSBA_{20}.pth'))

encoder_issba = StegaStampEncoder(
    secret_size=secret_size, 
    height=32, 
    width=32,
    in_channel=3).to(device)
savepath = os.path.join(exp_dir, 'encoder_decoder.pth')
state_pth = torch.load(savepath)
encoder_issba.load_state_dict(state_pth['encoder_state_dict']) 

model.eval(); encoder_issba.eval()
model.requires_grad_(False); encoder_issba.requires_grad_(False)
# prepare model to be trained B

# ----------------------------------------- 0.3 prepare data X_root X_questioned
ds_tr, ds_te, ds_x_root, ds_x_root_test, ds_x_q, ds_x_q_te = utils_data.prepare_CIFAR10_datasets(
    folder_=exp_dir, INITIAL_RUN=False)
assert len(ds_tr)==len(ds_x_root)+len(ds_x_q), f"wrong length, {len(ds_tr)} != {len(ds_x_root)}+{len(ds_x_q)}"
print(f'X_root: {len(ds_x_root)} samples, X_questioned: {len(ds_x_q)} samples')
bs_tr = 128

dl_root = DataLoader(
    dataset= ds_x_root,
    batch_size=bs_tr,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)

dl_x_q = DataLoader(
    dataset= ds_x_q,
    batch_size=bs_tr,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)
dl_te = DataLoader(
    dataset= ds_te,
    batch_size=bs_tr,
    shuffle=False,
    num_workers=0, 
    drop_last=False
)


# Now last_conv_output contains the output of the last convolutional layer
utils_attack.test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                        secret=secret, encoder=encoder_issba, device=device)

# prepare device


# training B
loader_root_iter = iter(dl_root)
loader_q_iter = iter(dl_x_q)
num_poison = int(ratio_poison*bs_tr)

# prepare B
B_theta = utils_attack.Encoder_no(); B_theta= B_theta.to(device)
optimizer = torch.optim.Adam(B_theta.parameters(), lr=lr_step1)

#TODO: add epochs
for epoch_ in range(epoch_step1):
    loss_sum = 0.0
    for i in range(max(len(dl_root), len(dl_x_q))):
        X_root, _ = get_next_batch(loader_root_iter, dl_root)
        X_q, _ = get_next_batch(loader_q_iter, dl_x_q)
        # X_root
        X_root, X_q = X_root.to(device), X_q.to(device)
        # poison X_q
        X_bd = copy.deepcopy(X_q)
        for xx in range(num_poison):
            X_bd[xx] = utils_attack.add_ISSBA_trigger(inputs=X_q[xx], secret=secret,
                                                            encoder=encoder_issba, device=device)
        X_q = X_bd[:num_poison]

        optimizer.zero_grad()
        B_root = B_theta(X_root)
        
        los_mse = utils_attack.reconstruction_loss(X_root, B_root) 
        loss = utils_defence.wasserstein_distance(model(B_root), model(X_q))
        loss.backward()
        optimizer.step()
        loss_sum+=loss.item()
    print(f'{loss_sum/len(dl_x_q): .2f}')
        # print(X_q.shape, X_root.shape)
        # fr1, fr2, fr3, fr4 = model.feature_(X_root)
        # fq1, fq2, fq3, fq4 = model.feature_(X_q)
        # loss = utils_defence.mmd_loss(fr1, fq1, sigma)+utils_defence.mmd_loss(fr2, fq2, sigma)+\
        #        utils_defence.mmd_loss(fr3, fq3, sigma)+utils_defence.mmd_loss(fr4, fq4, sigma)
        # loss2 = utils_defence.wasserstein_distance(X_root, X_q)

    
    




