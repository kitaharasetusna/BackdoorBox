# step3: pick up malicious samples

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
epoch_B = 10
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
encoder_issba.requires_grad_(False); model.requires_grad_(False)


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

ACC_, ASR_ = utils_attack.test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                        secret=secret, encoder=encoder_issba, device=device)

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
ds_whole_poisoned = utils_attack.CustomCIFAR10ISSBA_whole(ds_tr, ids_p, label_backdoor, secret, encoder_issba, device)


B_theta = utils_attack.Encoder_no(); B_theta= B_theta.to(device)
ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_tr,shuffle=True,num_workers=0,drop_last=False)
# TODO: change this
ds_sus = Subset(ds_whole_poisoned, ids_sus)
dl_sus = DataLoader(dataset= ds_sus,batch_size=bs_tr,shuffle=True,num_workers=0,drop_last=False)

loader_root_iter = iter(dl_root); loader_sus_iter = iter(dl_sus) 
optimizer = torch.optim.Adam(B_theta.parameters(), lr=lr_step1)

for epoch_ in range(epoch_B):
    loss_sum = 0.0; loss_wass_sum = 0.0; loss_mse_sum = 0.0
    for i in range(max(len(dl_root), len(dl_sus))):
        X_root, _ = get_next_batch(loader_root_iter, dl_root)
        X_q, _ = get_next_batch(loader_q_iter, dl_x_q)
        # X_root
        X_root, X_q = X_root.to(device), X_q.to(device)

        optimizer.zero_grad()
        B_root = B_theta(X_root)
        
        los_mse = utils_attack.reconstruction_loss(X_root, B_root) 
        loss_wass = utils_defence.wasserstein_distance(model(B_root), model(X_q))
        loss = los_mse + 10*loss_wass
        loss.backward()
        optimizer.step()
        loss_sum+=loss.item(); loss_mse_sum+=los_mse.item(); loss_wass_sum+=loss_wass.item()
    print(f'epoch: {epoch_}, loss: {loss_sum/len(dl_x_q): .2f}')
    print(f'loss mse: {loss_mse_sum/len(dl_x_q): .2f}')
    print(f'loss wass: {loss_wass_sum/len(dl_x_q): .2f}')
    if (epoch_+1)%5==0 or epoch_==epoch_B-1 or epoch_==0:
        utils_attack.test_asr_acc_ISSBA_gen(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                            B=B_theta, device=device)
        torch.save(B_theta.state_dict(), exp_dir+'/'+f'B_theta_{epoch_+1}.pth')

