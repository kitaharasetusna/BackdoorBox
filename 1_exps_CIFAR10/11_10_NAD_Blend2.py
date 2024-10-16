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
epoch_teacher = 10; lr_teacher = 1e-4 
epoch_NAD = 10; lr_NAD = 1e-4; power = 2.0
target_layers=['layer2', 'layer3', 'layer4']
beta=[500, 500, 500]
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

ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_tr,shuffle=True,num_workers=0,drop_last=True)
# ----------------------------------------- 1 train teacher model
train_teacher = False 
ACC, ASR = [], []
if train_teacher:
    

    model_teacher = copy.deepcopy(model)
    for param in model_teacher.parameters():
        param.requires_grad = True 
    optimizer = torch.optim.Adam(model_teacher.parameters(), lr=lr_teacher)    
    criterion = nn.CrossEntropyLoss()

    # --- training
    for epoch_ in range(epoch_teacher):
        for  X_root, Y_root in dl_root:
            X_root, Y_root = X_root.to(device), Y_root.to(device)
            optimizer.zero_grad()
            # make a forward pass
            Y_root_pred = model_teacher(X_root)
            # calculate the loss
            loss = criterion(Y_root_pred, Y_root)
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {epoch_+1}')
        if True:
            model_teacher.eval()
            ACC_, ASR_ =  utils_attack.test_asr_acc_blended(dl_te=dl_te, model=model_teacher,
                                label_backdoor=label_backdoor, pattern=pattern, device=device)
            ACC.append(ACC_); ASR.append(ASR_)
            with open(exp_dir+f'/10_NAD_train_teacher_model.pkl', 'wb') as f:
                pickle.dump({'ACC': ACC, 'ASR': ASR },f)
            torch.save(model_teacher.state_dict(), exp_dir+'/'+f'10_NAD_teacher_model_{epoch_+1}.pth')
            model_teacher.train() 

else:
    for param in model.parameters():
        param.requires_grad = True 
    model_teacher = copy.deepcopy(model)
    model_teacher.load_state_dict(torch.load(exp_dir+'/'+f'10_NAD_teacher_model_{9}.pth'))
    for param in model_teacher.parameters():
        param.requires_grad = False 
    
    # use NAD
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_NAD)    
    criterionAT = utils_defence.AT(power)

    ACC, ASR = [], []
    for epoch_ in range(epoch_NAD):
        for  X_root, Y_root in dl_root:
            X_root, Y_root = X_root.to(device), Y_root.to(device)
            optimizer.zero_grad()

            container = []
            def forward_hook(module, input, output):
                container.append(output)
            
            hook_list = []
            for name, module in model._modules.items():
                if name in target_layers:
                    hk = module.register_forward_hook(forward_hook)
                    hook_list.append(hk)

            for name, module in model_teacher._modules.items():
                if name in target_layers:
                    hk = module.register_forward_hook(forward_hook)
                    hook_list.append(hk)

            output_s = model(X_root)
            _ = model_teacher(X_root)

            for hk in hook_list:
                    hk.remove()

            loss = criterion(output_s, Y_root)
            for idx in range(len(beta)):
                loss = loss + criterionAT(container[idx], container[idx+len(beta)]) * beta[idx]   

            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch_+1}')
        if True:
            model.eval()
            ACC_, ASR_ =  utils_attack.test_asr_acc_blended(dl_te=dl_te, model=model,
                                label_backdoor=label_backdoor, pattern=pattern, device=device)
            ACC.append(ACC_); ASR.append(ASR_)
            with open(exp_dir+f'/10_NAD_learn_student_model.pkl', 'wb') as f:
                pickle.dump({'ACC': ACC, 'ASR': ASR },f)
            torch.save(model_teacher.state_dict(), exp_dir+'/'+f'10_NAD_student_model_{epoch_+1}.pth')
            model_teacher.train() 

    

    