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

# ----------------------------------------- 0.0 fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 0.1 configs:
exp_dir = '../experiments/exp6_FI_B/Blended2' 
secret_size = 20; label_backdoor = 6
bs_tr = 128; epoch_Blended = 20; lr_Blended = 1e-4
idx_blend = 656
lr_B = 1e-4;epoch_B = 10 
lr_ft = 1e-3
# ----------------------------------------- 0.2 dirs, load ISSBA_encoder+secret+model f'
# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)

device = torch.device("cuda:0")

model = core.models.ResNet(18); model = model.to(device)
model.load_state_dict(torch.load(exp_dir+'/step1_model_65.pth'))
criterion = nn.CrossEntropyLoss()

model.eval()
model.requires_grad_(False)


ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_CIFAR10_datasets_2(foloder=exp_dir,
                                load=True)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ds_q)}"

# ----------------------------------------- train model with ISSBA encoder
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)

pattern, _ = ds_te[idx_blend] #(3, 32, 32)

ds_questioned = utils_attack.CustomCIFAR10Blended(original_dataset=ds_tr, subset_indices=ids_q,
                trigger_indices=ids_p, label_bd=label_backdoor, pattern=pattern)
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


ACC_, ASR_ = utils_attack.test_asr_acc_blended(dl_te=dl_te, model=model,
                            label_backdoor=label_backdoor, pattern=pattern, device=device)

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
ds_whole_poisoned = utils_attack.CustomCIFAR10Blended_whole(ds_tr, ids_p, label_backdoor,
                    pattern=pattern)


B_theta = utils_attack.Encoder_no(); B_theta= B_theta.to(device)
ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_tr,shuffle=True,num_workers=0,drop_last=True)
ds_sus = Subset(ds_whole_poisoned, idx_sus)
dl_sus = DataLoader(dataset= ds_sus,batch_size=bs_tr,shuffle=True,num_workers=0,drop_last=True)

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
            # lpips_loss_op = loss_fn_alex(X_root, B_root)
            # loss_wass = utils_defence.wasserstein_distance(model(B_root), model(X_q))
            # loss = 20*los_mse + loss_wass
            # if epoch_<=20:
            #     loss = los_logits
            # else:
            loss = los_logits+los_mse
          
            loss.backward()
            optimizer.step()
            loss_mse_sum+=los_mse.item(); loss_logits_sum+=los_logits.item(); loss_inf_sum+=los_inf.item()
        print(f'epoch: {epoch_}')
        print(f'loss mse: {loss_mse_sum/len(dl_sus): .2f}')
        print(f'loss logits: {loss_logits_sum/len(dl_sus): .2f}')
        print(f'loss inf: {loss_inf_sum/len(dl_sus): .2f}')
        if (epoch_+1)%5==0 or epoch_==epoch_B-1 or epoch_==0:
            utils_attack.test_asr_acc_ISSBA_gen(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                B=B_theta, device=device)
            torch.save(B_theta.state_dict(), exp_dir+'/'+f'B_theta_{epoch_+1}.pth')
else:
    B_theta.eval()
    B_theta.requires_grad_(False) 
    for index in [100, 200, 300, 400, 500, 600]:
        with torch.no_grad(): 
            image_, _=ds_x_root[index]; image_c = copy.deepcopy(image_) 
            image_ = image_.to(device).unsqueeze(0); image = copy.deepcopy(image_)
            tensor_ori = copy.deepcopy(image_).to(device)
            image_ = utils_data.unnormalize(image_, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
            image_ = image_.squeeze().cpu().detach().numpy().transpose((1, 2, 0)) ;plt.imshow(image_);plt.savefig(exp_dir+f'/ori_{index}.pdf')

            encoded_image = utils_attack.add_blended_trigger(inputs=image_c, pattern=pattern, alpha=0.2)
            tensor_badnet = copy.deepcopy(encoded_image).to(device)
            encoded_image = utils_data.unnormalize(encoded_image, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) 
            issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
            plt.imshow(issba_image)
            plt.savefig(exp_dir+f'/Blended_{index}.pdf')

            encoded_image = B_theta(image)
            tensor_gen = copy.deepcopy(encoded_image).to(device)
            encoded_image = utils_data.unnormalize(encoded_image, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) 
            issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
            plt.imshow(issba_image)
            plt.savefig(exp_dir+f'/Gen_{index}.pdf') 

            
            
    for param in model.parameters():
        param.requires_grad = True 
    # TODO: move this to a single part
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_ft)
    criterion = nn.CrossEntropyLoss()
    utils_attack.test_acc(dl_te=dl_root, model=model, device=device)

    utils_attack.fine_tune_Blended2(dl_root=dl_root, model=model, label_backdoor=label_backdoor,
                                B=B_theta, device=device, dl_te=dl_te, pattern=pattern, 
                                epoch=10, optimizer=optimizer, criterion=criterion,
                                dl_sus=dl_sus, loader_root_iter=iter(dl_root), loader_sus_iter=iter(dl_sus))



