# train SAU on wanet
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

# ----------------------------------------- 1 fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 2 configs:
exp_dir = '../experiments/exp6_FI_B/BATT'; dataset = 'cifar10'; num_classes = 10
label_backdoor = 6
bs_tr = 128
lr_ft = 1e-4; epoch_SAU=10
beta_1 = 0.01; beta_2 = 1; trigger_norm = 0.2; norm_type = 'L_inf'
rotation = 16 
adv_lr = 0.2; adv_steps = 5; pgd_init = 'max'; outer_steps = 1
lmd_1 = 1; lmd_2 = 0.0; lmd_3 = 1
# ----------------------------------------- 3 load models 
os.makedirs(exp_dir, exist_ok=True)
device = torch.device("cuda:1")

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

# ----------------------------------------- 4 test model 
def comp_inf_norm(A, B):
    infinity_norm = torch.max(torch.abs(A - B))
    return infinity_norm

dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)

ACC_, ASR_ = utils_attack.test_asr_acc_batt(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                    rotation=rotation, device=device)

# ----------------------------------------- 5 prepare root data 
ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_tr,shuffle=True,num_workers=0,drop_last=True)

# ----------------------------------------- 6 train SAU 
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
    for index_img_bh, (images, labels) in enumerate(dl_root):
        images = images.to(device)
        labels = labels.to(device)

        max_eps = 1 - denormalization(images)
        min_eps = -denormalization(images)

        batch_pert = Shared_PGD_Attacker.attack(images, labels, max_eps, min_eps)

        if index_img_bh==0:
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

                    encoded_image = get_perturbed_image(image_c.to(device), pert=batch_pert[0])
                    tensor_gen = copy.deepcopy(encoded_image).to(device)
                    issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
                    plt.imshow(issba_image)
                    plt.savefig(exp_dir+f'/SAU{round}_{index}.pdf') 

                    norm_ori_bad = comp_inf_norm(tensor_ori, tensor_badnet)
                    norm_ori_gen = comp_inf_norm(tensor_ori, tensor_gen)
                    norm_bad_gen = comp_inf_norm(tensor_gen, tensor_badnet)
                    print(f'ori-bad: {norm_ori_bad}, ori-gen: {norm_ori_gen}, gen-bad: {norm_bad_gen}')

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
    
    ACC_, ASR_ = utils_attack.test_asr_acc_batt(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                    rotation=rotation, device=device) 