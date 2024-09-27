# 1. train BadNet on tinyImageNet 

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


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}')
    return val_loss/len(val_loader), 100 * correct / total

# ----------------------------------------- 1 fix seed -----------------------------------------
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 2 configs  ------------------------------------------
exp_dir = '../experiments/exp7_TinyImageNet/Blended' 
dataset = 'tiny_img'; num_classes = 200
secret_size = 20; label_backdoor = 6
bs_tr = 128
idx_blend = 656
bs_tr2 = 128 
lr_B = 1e-4; epoch_B = 100 
lr_ft = 1e-4; epoch_SAU = 10 
beta_1 = 0.01; beta_2 = 1; trigger_norm = 0.2; norm_type = 'L_inf'
rotation = 16 
adv_lr = 0.2; adv_steps = 5; pgd_init = 'max'; outer_steps = 1
lmd_1 = 1; lmd_2 = 0.0; lmd_3 = 1
alpha = 0.2
os.makedirs(exp_dir, exist_ok=True)

# ----------------------------------------- 3 load model ------------------------------------------
device = torch.device("cuda:0")
# Load a pretrained ResNet-18 model
model = models.resnet18(pretrained=True)
model = torchvision.models.get_model('resnet18', num_classes=200)
model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model.maxpool = nn.Identity()
model = model.to(device)
model.load_state_dict(torch.load(exp_dir+'/step1_model_1.pth'))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_ft)


# ----------------------------------------- 4 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_ImageNet_datasets(foloder=exp_dir,
                                load=False)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ids_q)}"

pattern, _ = ds_te[idx_blend] #(3, 32, 32)
ds_questioned = utils_attack.CustomCIFAR10Blended(original_dataset=ds_tr, subset_indices=ids_q,
                trigger_indices=ids_p, label_bd=label_backdoor, pattern=pattern, alpha=alpha)

dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
    num_workers=0,drop_last=False,
)
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)

ACC_, ASR_ = utils_attack.test_asr_acc_blended(dl_te=dl_te, model=model,
                        label_backdoor=label_backdoor, pattern=pattern, device=device, alpha=alpha) 

ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_tr,shuffle=True,num_workers=0,drop_last=True)
normalization = utils_defence.get_dataset_normalization(dataset)
denormalization = utils_defence.get_dataset_denormalization(normalization)

def get_perturbed_image(images, pert, train = True):
    images_wo_trans = denormalization(images) + pert
    images_with_trans = normalization(images_wo_trans)
    return images_with_trans

model_ref = copy.deepcopy(model)
model.eval()
model_ref.eval()

Shared_PGD_Attacker = utils_defence.Shared_PGD(model = model, 
                                model_ref = model_ref, 
                                beta_1 = beta_1, 
                                beta_2 = beta_2, 
                                norm_bound = trigger_norm, 
                                norm_type = norm_type, 
                                step_size = adv_lr, 
                                num_steps = adv_steps, 
                                init_type = pgd_init,
                                loss_func = torch.nn.CrossEntropyLoss(), 
                                pert_func = get_perturbed_image, 
                                verbose = True)
model.requires_grad_(True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_ft)
for round in range(epoch_SAU):
    for images, labels in dl_root:
        images = images.to(device)
        labels = labels.to(device)

        max_eps = 1 - denormalization(images)
        min_eps = -denormalization(images)

        batch_pert = Shared_PGD_Attacker.attack(images, labels, max_eps, min_eps)

        for _ in range(outer_steps):
            model.train()
            pert_image = get_perturbed_image(images, batch_pert.detach())
            concat_images = torch.cat([images, pert_image], dim=0)
            concat_logits = model.forward(concat_images)
            logits, per_logits = torch.split(concat_logits, images.shape[0], dim=0)
            # model.eval()
            
            logits_ref = model_ref(images)
            per_logits_ref = model_ref.forward(pert_image)

            # Get prediction
            ori_lab = torch.argmax(logits,axis = 1).long()
            ori_lab_ref = torch.argmax(logits_ref,axis = 1).long()

            pert_label = torch.argmax(per_logits, dim=1)
            pert_label_ref = torch.argmax(per_logits_ref, dim=1)

            success_attack = pert_label != labels
            success_attack_ref = pert_label_ref != labels
            success_attack_ref = success_attack_ref & (pert_label_ref != ori_lab_ref)
            common_attack = torch.logical_and(success_attack, success_attack_ref)
            shared_attack = torch.logical_and(common_attack, pert_label == pert_label_ref)

            # Clean loss
            loss_cl = F.cross_entropy(logits, labels, reduction='mean')
            
            # AT loss
            loss_at = F.cross_entropy(per_logits, labels, reduction='mean')
            
            # Shared loss
            potential_poison = success_attack_ref

            if potential_poison.sum() == 0:
                loss_shared = torch.tensor(0.0).to(device)
            else:
                one_hot = F.one_hot(pert_label_ref, num_classes=num_classes)
                
                neg_one_hot = 1 - one_hot
                neg_p = (F.softmax(per_logits, dim = 1)*neg_one_hot).sum(dim = 1)[potential_poison]
                pos_p = (F.softmax(per_logits, dim = 1)*one_hot).sum(dim = 1)[potential_poison]
                
                # clamp the too small values to avoid nan and discard samples with p<1% to be shared
                # Note: The below equation combine two identical terms in math. Although they are the same in math, they are different in implementation due to the numerical issue. 
                #       Combining them can reduce the numerical issue.

                loss_shared = (-torch.sum(torch.log(1e-6 + neg_p.clamp(max = 0.999))) - torch.sum(torch.log(1 + 1e-6 - pos_p.clamp(min = 0.001))))/2
                loss_shared = loss_shared/images.shape[0]
            
            optimizer.zero_grad()

            loss = lmd_1*loss_cl + lmd_2* loss_at + lmd_3*loss_shared
            loss.backward()
            optimizer.step()

            # delete the useless variable to save memory
            del logits, logits_ref, per_logits, per_logits_ref, loss_cl, loss_at, loss_shared, loss
    utils_attack.test_asr_acc_blended(dl_te=dl_te, model=model,
                        label_backdoor=label_backdoor, pattern=pattern, device=device, alpha=alpha)      

    
