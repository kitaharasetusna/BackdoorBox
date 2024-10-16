# step3 and step4: train B theta ;break up suprious relationships

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
import lpips
sys.path.append('..')
import core
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator
from myutils import utils_data, utils_attack, utils_defence
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchvision import models

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
lr_B = 1e-2;epoch_B = 30 
train_B = False 
exp_dir = '../experiments/exp7_TinyImageNet/SIG' 
dataset = 'tiny_img'
label_backdoor = 6
bs_tr = 128; epoch_SIG = 100; lr_SIG = 1e-3
bs_tr2 = 128; bs_tr_s = 45 
sig_delta = 40; sig_f = 6
lr_ft = 1e-4; epoch_root = 10
normalization = utils_defence.get_dataset_normalization(dataset)
denormalization = utils_defence.get_dataset_denormalization(normalization)
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
model.load_state_dict(torch.load(exp_dir+'/step1_model_10.pth'))
criterion = nn.CrossEntropyLoss()

model.eval()
model.requires_grad_(False)

# ----------------------------------------- 0.3 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_ImageNet_datasets_SIG_2(foloder=exp_dir,
                                load=True, target_label=label_backdoor)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ids_q)}"
# ----------------------------------------- train model with ISSBA encoder
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)
ds_questioned = utils_attack.CustomCIFAR10SIG(original_dataset=ds_tr, subset_indices=ids_q+ids_root,
                                               trigger_indices=ids_p, label_bd=label_backdoor,
                                               delta=sig_delta, frequency=sig_f,
                                               norm=normalization, denorm=denormalization)
dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
    num_workers=0,drop_last=False,
)
ACC_, ASR_ = utils_attack.test_asr_acc_sig(dl_te=dl_te, model=model,
                                                   label_backdoor=label_backdoor,
                                                   delta=sig_delta, freq=sig_f, device=device,
                                                   norm=normalization, denorm=denormalization)

if train_B:
    with open(exp_dir+'/idx_suspicious2.pkl', 'rb') as f:
        idx_sus = pickle.load(f)
else:
    with open(exp_dir+'/idx_suspicious2.pkl', 'rb') as f:
        idx_sus = pickle.load(f)
print(len(idx_sus))
TP, FP = 0.0, 0.0
for s in idx_sus:
    if s in ids_p:
        TP+=1
    else:
        FP+=1
print(TP/(TP+FP))

# ----------------------------------------- 1 train B_theta  
# prepare B
ds_whole_poisoned = utils_attack.CustomCIFAR10SIG_whole(ds_tr, ids_p, label_backdoor, sig_delta, sig_f,
                                                        norm=normalization, denorm=denormalization)

# B_theta = utils_attack.FixedSTN(input_channels=3, device=device)
B_theta = utils_attack.Encoder_no()
B_theta= B_theta.to(device)
ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_tr_s,shuffle=True,num_workers=0,drop_last=True)
# TODO: change this
ds_sus = Subset(ds_whole_poisoned, idx_sus)
dl_sus = DataLoader(dataset= ds_sus,batch_size=bs_tr_s,shuffle=True,num_workers=0,drop_last=True)

# for debugging step 2
# for idx_s in range(5):
#     # TODO: in BvB debug print, use the same index
#     image, label = ds_sus[idx_s]
#     image = image.to(device).unsqueeze(0) #(1, 3, 64, 64)
#     image = utils_data.unnormalize(image, mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
#     image = image.squeeze().cpu().numpy().transpose((1, 2, 0))
#     plt.imshow(image); plt.savefig(exp_dir+f'/step4_debug_find_sus_{idx_s}.pdf')

# # for debugging step 2
# for idx_root in range(5):
#     # TODO: in BvB debug print, use the same index
#     image, label = ds_x_root[idx_root]
#     image = image.to(device).unsqueeze(0) #(1, 3, 64, 64)
#     image = utils_data.unnormalize(image, mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
#     image = image.squeeze().cpu().numpy().transpose((1, 2, 0))
#     plt.imshow(image); plt.savefig(exp_dir+f'/step4_debug_root_{idx_root}.pdf')

loader_root_iter = iter(dl_root); loader_sus_iter = iter(dl_sus) 
optimizer = torch.optim.Adam(B_theta.parameters(), lr=lr_B)
print(len(ds_sus), len(ds_x_root))


def relu_(x, threshold=0.5):
    if x>threshold:
        return x 
    else:
        return torch.tensor(0.0)

if train_B:
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
            
            #TODO: compare the mse we have and the per-sample mse sig has
            loss = los_logits+10*los_mse
            # loss = los_logits
          
            loss.backward()
            optimizer.step()
            loss_mse_sum+=los_mse.item(); loss_logits_sum+=los_logits.item(); loss_inf_sum+=los_inf.item()
        print(f'epoch: {epoch_}')
        print(f'loss mse: {loss_mse_sum/len(ds_sus): .2f}')
        print(f'loss logits: {loss_logits_sum/len(dl_sus): .2f}')
        print(f'loss inf: {loss_inf_sum/len(dl_sus): .2f}')
        if True:
            utils_attack.test_asr_acc_BATT_gen_residual(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                B=B_theta, device=device) 
            torch.save(B_theta.state_dict(), exp_dir+'/'+f'B_theta_{epoch_+1}.pth')
else:
    pth_path = exp_dir+'/'+f'B_theta_{11}.pth'
    B_theta.load_state_dict(torch.load(pth_path))
    B_theta.eval()
    B_theta.requires_grad_(False) 
    utils_attack.test_asr_acc_BATT_gen_residual(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                B=B_theta, device=device) 
    # for index in [100, 200, 300, 400, 500, 600]:
    #     with torch.no_grad(): 
    #         image_, _=ds_x_root[index]; image_single = copy.deepcopy(image_) 
    #         image_ = image_.to(device).unsqueeze(0); image_batch = copy.deepcopy(image_)
    #         tensor_ori = copy.deepcopy(image_).to(device)
    #         image_ = utils_data.unnormalize(image_, mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    #         image_ = image_.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
    #         plt.imshow(image_);plt.savefig(exp_dir+f'/ori_{index}.pdf')
            
    #         image_sig = utils_attack.add_SIG_trigger(inputs=image_single, delta=sig_delta, frequency=sig_f, norm=normalization,
    #                                                      denorm=denormalization)
    #         tensor_sig = copy.deepcopy(image_sig).to(device)
    #         image_sig = image_sig.to(device).unsqueeze(0)
    #         image_sig = utils_data.unnormalize(image_sig, mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    #         image_sig = image_sig.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
    #         plt.imshow(image_sig)
    #         plt.savefig(exp_dir+f'/SIG_{index}.pdf')

    #         image_bvb = B_theta(image_batch)
    #         tensor_bvb = copy.deepcopy(image_bvb).to(device)
    #         # TODO: add un-normalize
    #         image_bvb = image_bvb.to(device)
    #         image_bvb = utils_data.unnormalize(image_bvb, mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    #         image_bvb = image_bvb.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
    #         plt.imshow(image_bvb)
    #         plt.savefig(exp_dir+f'/generated_{index}.pdf')

    #         norm_ori_bad = comp_inf_norm(tensor_ori, tensor_sig)
    #         norm_ori_gen = comp_inf_norm(tensor_ori, tensor_bvb)
    #         norm_bad_gen = comp_inf_norm(tensor_bvb, tensor_sig)
    #         print(f'ori-bad: {norm_ori_bad}, ori-gen: {norm_ori_gen}, gen-bad: {norm_bad_gen}')

    model.train() 
    for param in model.parameters():
        param.requires_grad = True 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_ft)
    criterion = nn.CrossEntropyLoss()

    utils_attack.fine_tune_SIG_tiny_Img(dl_root=dl_root, model=model, label_backdoor=label_backdoor,
                                B=B_theta, device=device, dl_te=dl_te, delta=sig_delta,
                                freq=sig_f, 
                                epoch=10, optimizer=optimizer, criterion=criterion,
                                dl_sus=dl_sus, loader_root_iter=iter(dl_root), loader_sus_iter=iter(dl_sus)
                                ,norm=normalization, denorm=denormalization)


