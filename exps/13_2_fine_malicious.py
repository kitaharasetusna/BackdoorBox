# step2: pick up malicious samples

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
import pickle
sys.path.append('..')
import core
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator
from myutils import utils_data, utils_attack, utils_defence
import matplotlib.pyplot as plt
from torch.utils.data import Subset



# ----------------------------------------- 1 fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 2 configs:
exp_dir = '../experiments/exp6_FI_B/SIG' 
label_backdoor = 6
bs_tr = 128; epoch_SIG = 100; lr_SIG = 1e-4
sig_delta = 40; sig_f = 6

# -----------------------------------------3 dirs, load model
os.makedirs(exp_dir, exist_ok=True)

device = torch.device("cuda:0")

model = core.models.ResNet(18); model = model.to(device)
model.load_state_dict(torch.load(exp_dir+'/step1_model_1.pth'))
criterion = nn.CrossEntropyLoss()

model.eval()

# ----------------------------------------- 0.3 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_CIFAR10_datasets_SIG(foloder=exp_dir,
                                load=True, target_label=label_backdoor)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ds_q)}"
# ----------------------------------------- train model with ISSBA encoder
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)
# TODO: change this to SIG attack
ds_questioned = utils_attack.CustomCIFAR10SIG(original_dataset=ds_tr, subset_indices=ids_q+ids_root,
                                               trigger_indices=ids_p, label_bd=label_backdoor,
                                               delta=sig_delta, frequency=sig_f)
dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
    num_workers=0,drop_last=False,
)

ds_x_q2 = Subset(ds_tr, ids_q)
dl_x_q2 = DataLoader(
    dataset= ds_x_q2,
    batch_size=bs_tr,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)

# ----------------------------------------- 1.1 compute average FI in root data
# TODO:

# ----------------------------------------- 1.2 pick up X suspicious
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
        print(idx_ori, 0, loss)
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
    top_10_percent_count = max(1, int(len(sorted_items) * 1 // 300))
    ids_suspicious = [item[0] for item in sorted_items[:top_10_percent_count]]
    with open(exp_dir+'/idx_suspicious.pkl', 'wb') as f:
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