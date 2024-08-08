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
exp_dir = '../experiments/exp6_FI_B/ISSBA' 
secret_size = 20; label_backdoor = 6 
bs_tr = 128

# ----------------------------------------- 0.2 dirs, load ISSBA_encoder+secret+model f'
# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)

device = torch.device("cuda:0")
# Load ISSBA encoder
encoder_issba = StegaStampEncoder(
    secret_size=secret_size, 
    height=32, 
    width=32,
    in_channel=3).to(device)
savepath = os.path.join(exp_dir, 'encoder_decoder.pth'); state_pth = torch.load(savepath)
encoder_issba.load_state_dict(state_pth['encoder_state_dict']) 
secret = torch.FloatTensor(np.random.binomial(1, .5, secret_size).tolist()).to(device)
model = core.models.ResNet(18); model = model.to(device)
model.load_state_dict(torch.load(exp_dir+'/model_1.pth'))
criterion = nn.CrossEntropyLoss()

encoder_issba.eval(); model.eval()
encoder_issba.requires_grad_(False)


# ----------------------------------------- 0.3 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_CIFAR10_datasets_2(foloder=exp_dir,
                                load=True)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ds_q)}"

ds_questioned = utils_attack.CustomCIFAR10ISSBA(
    ds_tr, ids_q, ids_p, label_backdoor, secret, encoder_issba, device)
# dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
#     num_workers=0,drop_last=False,
# )
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)

# ACC_, ASR_ = utils_attack.test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
#                                         secret=secret, encoder=encoder_issba, device=device)
ds_x_q2 = Subset(ds_tr, ids_q)
dl_x_q2 = DataLoader(
    dataset= ds_x_q2,
    batch_size=bs_tr,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)
avg_trace_fim, avg_trace_fim_bd, avg_loss_cln, avg_loss_bd = get_train_fim_ISSBA(model=model, dl_train=dl_x_q2,
                                                          encoder=encoder_issba, secret=secret,
                                                          ratio_poison=0.1, bs_tr=bs_tr,
                                                          device=device)
print(avg_trace_fim, 'clean')
print(avg_trace_fim_bd, 'bd')
# ----------------------------------------- 1.1 compute average FI in root data
# TODO:

# ----------------------------------------- 1.2 pick up X suspicious
from collections import defaultdict
pick_upX = True 
if pick_upX:
    data = defaultdict(int)
    for i in range(len(ds_questioned)):
        image, label = ds_questioned[i]  
        image, label = image.unsqueeze(0), torch.tensor(label).unsqueeze(0)
        fi_value = utils_defence.compute_fisher_information(model=model, images=image, 
                                                            labels=label, criterion=criterion,
                                                            device=device, loss_=False)  # 计算FI值
        idx_ori = ds_questioned.original_dataset.indices[i]
        data[idx_ori] = fi_value
        with open(exp_dir+'step_2_X_suspicious_dict.pkl', 'wb') as f:
            pickle.dump(data, f)
    sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=True)
    top_10_percent_count = max(1, len(sorted_items) * 10 // 100)
    ids_suspicious = [item[0] for item in sorted_items[:top_10_percent_count]]
 
    

    with open(exp_dir+'step_2_X_suspicious.pkl', 'wb') as f:
        pickle.dump(ids_suspicious, f)
else: 
    with  open(exp_dir+'step_2_X_suspicious.pkl', 'rb') as f:
        ids_suspicious = pickle.load(f)
        print(len(ids_suspicious))
        print(len(ids_p))
    
    TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0
    for s in ids_suspicious:
        if s in ids_p:
            TP+=1
        else:
            FP+=1
    FN = len(ids_p)-TP if TP< len(ids_p) else 0
    TN = len(ds_questioned)-FP
    F1 = 2*TP/(2*TP+FP+FN)
    recall = TP/(TP+FN)
    print(F1, recall)