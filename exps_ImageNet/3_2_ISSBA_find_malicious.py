# step2: pick up malicious samples

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
sys.path.append('..')
import core
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator
from myutils import utils_data, utils_attack, utils_defence
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchvision import models


# ----------------------------------------- 1 fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 2 configs:
exp_dir = '../experiments/exp7_TinyImageNet/ISSBA' 
secret_size = 20; label_backdoor = 6 
bs_tr = 512
os.makedirs(exp_dir, exist_ok=True)
get_smaller_idx = True 
# -----------------------------------------3 dirs, load model
os.makedirs(exp_dir, exist_ok=True)

device = torch.device("cuda:0")
# Load a pretrained ResNet-18 model
model = models.resnet18(pretrained=True)
model = torchvision.models.get_model('resnet18', num_classes=200)
model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model.maxpool = nn.Identity()
model = model.to(device)
model.load_state_dict(torch.load(exp_dir+f'/step1_model_detector.pth'))
encoder_issba = StegaStampEncoder(
    secret_size=secret_size, 
    height=64, 
    width=64,
    in_channel=3).to(device)
savepath = os.path.join(exp_dir, 'encoder_decoder.pth')
state_pth = torch.load(savepath)
encoder_issba.load_state_dict(state_pth['encoder_state_dict']) 
criterion = nn.CrossEntropyLoss()

model.eval()

# ----------------------------------------- 0.3 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln, ids_noise = utils_data.prepare_ImageNet_datasets_WaNet(foloder=exp_dir,
                                load=True)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}, noise: {len(ids_noise)}")

# ----------------------------------------- 5 train model 
secret = torch.FloatTensor(np.random.binomial(1, .5, secret_size).tolist()).to(device)
ds_questioned = utils_attack.CustomCIFAR10ISSBA(
    ds_tr, ids_q, ids_p, label_backdoor, secret, encoder_issba, device)
dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
    num_workers=0,drop_last=False,
)
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)
utils_attack.test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                        secret=secret, encoder=encoder_issba, device=device)
# ----------------------------------------- 5. pick up malicious samples 
from collections import defaultdict
pick_upX = False 
if pick_upX:
    data = defaultdict(int)
    print("start FI collecting")
    for i in range(len(ds_questioned)):
        image, label = ds_questioned[i]  
        image, label = image.unsqueeze(0), torch.tensor(label).unsqueeze(0)
        loss = utils_defence.compute_loss(model=model, images=image, labels=label,
                                          criterion=criterion, device=device) 
        idx_ori = ds_questioned.original_dataset.indices[i]
        if idx_ori in ids_p:
            tag = 'poison'
        else:
            tag = 'clean'
        if tag == 'poison':
            print(idx_ori, tag, loss)
        data[idx_ori] = (0, loss)
    print("finish FI collecting")
    with open(exp_dir+'/step_2_X_suspicious_dict.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("finish FI saving")
else: 
    # with  open(exp_dir+'/step_2_X_suspicious.pkl', 'rb') as f:
    #     ids_suspicious= pickle.load(f)
    # print(len(ids_suspicious))
    with open(exp_dir+'/step_2_X_suspicious_dict.pkl', 'rb') as f:
        data = pickle.load(f)
    # Filter keys with values greater than 27
    sorted_items = sorted(data.items(), key=lambda item: item[1][1])
    if not get_smaller_idx:
        top_10_percent_count = max(1, int(len(sorted_items) * 1 // 100))
        ids_suspicious = [item[0] for item in sorted_items[:top_10_percent_count]]
        with open(exp_dir+'/idx_suspicious.pkl', 'wb') as f:
            pickle.dump(ids_suspicious, f)
    else:
        top_10_percent_count = max(1, int(len(sorted_items) * 1 // 1000))
        ids_suspicious = [item[0] for item in sorted_items[:top_10_percent_count]]
        with open(exp_dir+'/idx_suspicious2.pkl', 'wb') as f:
            pickle.dump(ids_suspicious, f)
    
    
    TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0
    for s in ids_suspicious:
        if s in ids_p:
            TP+=1
        else:
            FP+=1
    FN = len(ids_p)-TP if TP< len(ids_p) else 0
    TN = len(ds_questioned)-FP
    F1 = 2*TP/(2*TP+FP+FN)
    precision = TP/(TP+FP)
    print(F1, precision)