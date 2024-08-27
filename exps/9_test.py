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
from torch.utils.data import Subset

# ----------------------------------------- 0.0 fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 0.1 configs:
exp_dir = '../experiments/exp6_FI_B/WaNet' 
label_backdoor = 6
bs_tr = 128; epoch_WaNet = 20; lr_WaNet = 1e-4
# ----------------------------------------- 0.2 dirs, load ISSBA_encoder+secret
# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)

device = torch.device("cuda:0")

model = core.models.ResNet(18); model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr_WaNet)
criterion = nn.CrossEntropyLoss()

# ----------------------------------------- 0.3 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln, ids_noise = utils_data.prepare_CIFAR10_datasets_3(foloder=exp_dir,
                                load=False)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}, noise: {len(ids_noise)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ds_q)}"

dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)

load_grid = False 
if not load_grid:
    identity_grid,noise_grid=utils_attack.gen_grid(32,4)
    torch.save(identity_grid, exp_dir+'/step1_ResNet-18_CIFAR-10_WaNet_identity_grid.pth')
    torch.save(noise_grid, exp_dir+'/step1_ResNet-18_CIFAR-10_WaNet_noise_grid.pth')
else:
    identity_grid = torch.load(exp_dir+'/step1_ResNet-18_CIFAR-10_WaNet_identity_grid.pth')
    noise_grid = torch.load(exp_dir+'/step1_ResNet-18_CIFAR-10_WaNet_noise_grid.pth')

print(identity_grid.shape, noise_grid.shape)
    
ds_questioned = utils_attack.CustomCIFAR10WaNet(
    ds_tr, ids_q, ids_p, noise_ids=ids_noise, label_bd=label_backdoor, identity_grid=identity_grid, noise_grid=noise_grid)
dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
    num_workers=0,drop_last=False,
)
ds_x_root = Subset(ds_tr, ids_root)
for index in [100, 200, 300, 400, 500, 600]:
        with torch.no_grad(): 
            image_, _=ds_x_root[index]; image_c = copy.deepcopy(image_) 
            image_ = image_.to(device).unsqueeze(0); image = copy.deepcopy(image_)
            tensor_ori = copy.deepcopy(image_).to(device)
            image_ = utils_data.unnormalize(image_, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
            image_ = image_.squeeze().cpu().detach().numpy().transpose((1, 2, 0)) ;plt.imshow(image_);plt.savefig(exp_dir+f'/ori_{index}.pdf')

            encoded_image = utils_attack.add_WaNet_trigger(inputs=image_c, identity_grid=identity_grid,
                                                           noise_grid=noise_grid)
            encoded_image = utils_data.unnormalize(encoded_image, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) 
            issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
            plt.imshow(issba_image)
            plt.savefig(exp_dir+f'/WaNet_{index}.pdf')
