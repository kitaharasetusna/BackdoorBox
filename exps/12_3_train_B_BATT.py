# step3 and step4: train B theta ;break up suprious relationships

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
import lpips
sys.path.append('..')
import core
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator
from myutils import utils_data, utils_attack, utils_defence
import matplotlib.pyplot as plt
from torch.utils.data import Subset

def comp_inf_norm(A, B):
    infinity_norm = torch.max(torch.abs(A - B))
    return infinity_norm

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
exp_dir = '../experiments/exp6_FI_B/BATT' 
label_backdoor = 6
bs_tr = 128; epoch_BATT = 100; lr_BATT = 1e-3
rotation = 16 
bs_tr2 = 32
lr_B = 1e-2;epoch_B = 100 
lr_ft = 1e-5
# ----------------------------------------- 0.2 dirs, load ISSBA_encoder+secret+model f'
# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)

device = torch.device("cuda:0")

model = core.models.ResNet(18); model = model.to(device)
model.load_state_dict(torch.load(exp_dir+'/step1_model_20.pth'))
criterion = nn.CrossEntropyLoss()

model.eval()
model.requires_grad_(False)


ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_CIFAR10_datasets_batt(foloder=exp_dir,
                                load=True)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ds_q)}"

# ----------------------------------------- train model with ISSBA encoder
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)
ds_questioned = utils_attack.CustomCIFAR10BATT(original_dataset=ds_tr, subset_indices=ids_q,
                                               trigger_indices=ids_p, label_bd=label_backdoor, roation=rotation)
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


ACC_, ASR_ = utils_attack.test_asr_acc_batt(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                    rotation=rotation, device=device)

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
# TODO: change this to blended_whole
ds_whole_poisoned = utils_attack.CustomCIFAR10BATT_whole(original_dataset=ds_tr,
                    trigger_indices=ids_p, label_bd=label_backdoor, rotation=rotation)

# B_theta = utils_attack.EncoderWithFixedTransformation(input_channels=3, device=device); 
B_theta = utils_attack.FixedSTN(input_channels=3, device=device)
B_theta= B_theta.to(device)
ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_tr2,shuffle=True,num_workers=0,drop_last=True)
ds_sus = Subset(ds_whole_poisoned, idx_sus)
dl_sus = DataLoader(dataset= ds_sus,batch_size=bs_tr2,shuffle=True,num_workers=0,drop_last=True)

loader_root_iter = iter(dl_root); loader_sus_iter = iter(dl_sus) 
optimizer = torch.optim.Adam(B_theta.parameters(), lr=lr_B)

train_B = False 

def relu_(x, threshold=0.5):
    if x>threshold:
        return x 
    else:
        return torch.tensor(0.0)

if train_B:
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    for epoch_ in range(epoch_B):
        loss_mse_sum = 0.0; loss_logits_sum = 0.0; loss_inf_sum = 0.0
        for i in range(max(len(dl_root), len(dl_sus))):
            X_root, _ = get_next_batch(loader_root_iter, dl_root)
            X_q, _ = get_next_batch(loader_sus_iter, dl_sus)
            # X_root
            X_root, X_q = X_root.to(device), X_q.to(device)

            optimizer.zero_grad()
            B_root = B_theta(X_root)
            
            los_mse = utils_attack.reconstruction_loss(X_root, B_root) 
            logits_root = model(B_root); logits_q = model(X_q)
            los_logits = F.kl_div(F.log_softmax(logits_root, dim=1), F.softmax(logits_q, dim=1), reduction='batchmean')
            los_inf =  torch.mean(torch.max(torch.abs(B_root - X_root), dim=1)[0])
            loss = los_logits+los_mse
          
            loss.backward()
            optimizer.step()
            loss_mse_sum+=los_mse.item(); loss_logits_sum+=los_logits.item(); loss_inf_sum+=los_inf.item()
        print(f'epoch: {epoch_}')
        print(f'loss mse: {loss_mse_sum/len(dl_sus): .2f}')
        print(f'loss logits: {loss_logits_sum/len(dl_sus): .2f}')
        print(f'loss inf: {loss_inf_sum/len(dl_sus): .2f}')
        if (epoch_+1)%5==0 or epoch_==epoch_B-1 or epoch_==0:
            utils_attack.test_asr_acc_BATT_gen(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                B=B_theta, device=device)
            torch.save(B_theta.state_dict(), exp_dir+'/'+f'B_theta_{epoch_+1}.pth')
else:
    pth_path = exp_dir+'/'+f'B_theta_{5}.pth'
    B_theta.load_state_dict(torch.load(pth_path))
    B_theta.eval()
    B_theta.requires_grad_(False) 
    for index in [100, 200, 300, 400, 500, 600]:
        with torch.no_grad(): 
            image_, _=ds_x_root[index]; image_c = copy.deepcopy(image_) 
            image_ = image_.to(device).unsqueeze(0); image = copy.deepcopy(image_)
            tensor_ori = copy.deepcopy(image_).to(device)
            image_ = image_.squeeze().cpu().detach().numpy().transpose((1, 2, 0)) ;plt.imshow(image_);plt.savefig(exp_dir+f'/ori_{index}.pdf')

            encoded_image = utils_attack.add_batt_trigger(inputs=image_c, rotation=rotation)
            tensor_badnet = copy.deepcopy(encoded_image).to(device)
            issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
            plt.imshow(issba_image)
            plt.savefig(exp_dir+f'/BATT_{index}.pdf')

            encoded_image = B_theta(image)
            tensor_gen = copy.deepcopy(encoded_image).to(device)
            issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
            plt.imshow(issba_image)
            plt.savefig(exp_dir+f'/Gen_{index}.pdf') 

            norm_ori_bad = comp_inf_norm(tensor_ori, tensor_badnet)
            norm_ori_gen = comp_inf_norm(tensor_ori, tensor_gen)
            norm_bad_gen = comp_inf_norm(tensor_gen, tensor_badnet)
            print(f'ori-bad: {norm_ori_bad}, ori-gen: {norm_ori_gen}, gen-bad: {norm_bad_gen}')
    
    for param in model.parameters():
        param.requires_grad = True 
    # TODO: move this to a single part
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_ft)
    criterion = nn.CrossEntropyLoss()
    utils_attack.test_acc(dl_te=dl_root, model=model, device=device)

    utils_attack.fine_tune_BATT(dl_root=dl_root, model=model, label_backdoor=label_backdoor,
                                B=B_theta, device=device, dl_te=dl_te, rotation=rotation, 
                                epoch=20, optimizer=optimizer, criterion=criterion,
                                dl_sus=dl_sus, loader_root_iter=iter(dl_root), loader_sus_iter=iter(dl_sus))



