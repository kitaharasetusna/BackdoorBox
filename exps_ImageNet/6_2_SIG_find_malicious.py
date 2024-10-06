# step2: pick up malicious samples
# TODO: fix bug for SIG add 

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
exp_dir = '../experiments/exp7_TinyImageNet/SIG' 
dataset = 'tiny_img'
label_backdoor = 6
bs_tr = 128; epoch_SIG = 100; lr_SIG = 1e-4
sig_delta = 40; sig_f = 6
os.makedirs(exp_dir, exist_ok=True)
get_smaller_idx = False 
# -----------------------------------------3 dirs, load model

device = torch.device("cuda:0")
# Load a pretrained ResNet-18 model
model = models.resnet18(pretrained=True)
model = torchvision.models.get_model('resnet18', num_classes=200)
model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model.maxpool = nn.Identity()
model = model.to(device)
model.load_state_dict(torch.load(exp_dir+'/'+f'step1_model_detector.pth'))
criterion = nn.CrossEntropyLoss()

model.eval()
normalization = utils_defence.get_dataset_normalization(dataset)
denormalization = utils_defence.get_dataset_denormalization(normalization)
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
ds_questioned = utils_attack.CustomCIFAR10SIG(original_dataset=ds_tr, subset_indices=ids_q,
                                               trigger_indices=ids_p, label_bd=label_backdoor,
                                               delta=sig_delta, frequency=sig_f, norm=normalization,
                                               denorm=denormalization)
ACC_, ASR_ = utils_attack.test_asr_acc_sig(dl_te=dl_te, model=model,
                                                   label_backdoor=label_backdoor,
                                                   delta=sig_delta, freq=sig_f, device=device, 
                                                   norm=normalization, denorm=denormalization)
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
        print(idx_ori, tag, loss)
        data[idx_ori] = (0, loss)
    print("finish FI collecting")
    with open(exp_dir+'/step_2_X_suspicious_dict.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("finish FI saving")
else: 
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
        top_10_percent_count = max(1, int(len(sorted_items) * 1 // 500))
        ids_suspicious = [item[0] for item in sorted_items[:top_10_percent_count]]
        with open(exp_dir+'/idx_suspicious2.pkl', 'wb') as f:
            pickle.dump(ids_suspicious, f)
    
    
    TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0
    for s in ids_suspicious:
        if s in ids_p:
            TP+=1
        else:
            FP+=1
    FN = len(ids_p)-TP  
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print(f'recall: {recall}', f'precision: {precision}')

test_idx_suspicious = ids_suspicious[:5]
ds_whole_poisoned = utils_attack.CustomCIFAR10SIG_whole(ds_tr, ids_p, label_backdoor, sig_delta, sig_f, norm=normalization,
                                                        denorm=denormalization)
for idx_s in test_idx_suspicious:
    image, label = ds_whole_poisoned[idx_s]
    
    image = image.to(device).unsqueeze(0) #(1, 3, 64, 64)
    image = utils_data.unnormalize(image, mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    image = image.squeeze().cpu().numpy().transpose((1, 2, 0))
    plt.imshow(image); plt.savefig(exp_dir+f'/ste2_debug_find_sus_{idx_s}.pdf')

    image_clean, label_clean = ds_tr[idx_s]
    image_clean = image_clean.to(device).unsqueeze(0) #(1, 3, 64, 64)
    image_clean = utils_data.unnormalize(image_clean, mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    image_clean = image_clean.squeeze().cpu().numpy().transpose((1, 2, 0))
    plt.imshow(image_clean); plt.savefig(exp_dir+f'/ste2_debug_find_clean_{idx_s}.pdf')

