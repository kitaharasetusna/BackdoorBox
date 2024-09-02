# step1: train model

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
sys.path.append('..')
import core
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator
from myutils import utils_data, utils_attack, utils_defence
import matplotlib.pyplot as plt
from PIL import Image


# ----------------------------------------- 0.0 fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)


# ----------------------------------------- 0.0.1 load pattern 
def load_patter(path_):
    img = Image.open(path_).convert('RGB')
    img = img.resize((32, 32))
    trigger = np.array(img).astype(np.float32) / 255.0
    return torch.tensor(trigger).permute(2, 0, 1)  # Convert to CHW format

# ----------------------------------------- 0.1 configs:
exp_dir = '../experiments/exp6_FI_B/Blended2' 
secret_size = 20; label_backdoor = 6
bs_tr = 128; epoch_Blended = 100; lr_Blended = 1e-4
idx_blend = 656
# ----------------------------------------- 0.2 dirs, load ISSBA_encoder+secret
# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)
device = torch.device("cuda:0")

model = core.models.ResNet(18); model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_Blended)
criterion = nn.CrossEntropyLoss()

# ----------------------------------------- 0.3 prepare data X_root X_questioned
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
# pattern = load_patter(exp_dir+'/hello_kitty.jpg')

#TODO: change this to customCIFAR10Blended
ds_questioned = utils_attack.CustomCIFAR10Blended(original_dataset=ds_tr, subset_indices=ids_q,
                trigger_indices=ids_p, label_bd=label_backdoor, pattern=pattern, alpha=0.2)
dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
    num_workers=0,drop_last=False,
)


model.train()
ACC = []; ASR= []
for epoch_ in range(epoch_Blended):
    for inputs, targets in dl_x_q:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # make a forward pass
        outputs = model(inputs)
        # calculate the loss
        loss = criterion(outputs, targets)
        # do a backwards pass
        loss.backward()
        # perform a single optimization step
        optimizer.step()
    print(f'epoch: {epoch_+1}')
    # if (epoch_+1)%5==0 or epoch_==0 or epoch_==epoch_Blended-1:
    if True:
        model.eval()
        # TODO: change this to test_asr_acc_blended
        ACC_, ASR_ = utils_attack.test_asr_acc_blended(dl_te=dl_te, model=model,
                        label_backdoor=label_backdoor, pattern=pattern, device=device)
        ACC.append(ACC_); ASR.append(ASR_)
        torch.save(model.state_dict(), exp_dir+'/'+f'step1_model_{epoch_+1}.pth')
        with open(exp_dir+f'/step1_train_blended.pkl', 'wb') as f:
            pickle.dump({'ACC': ACC, 'ASR': ASR },f)
        model.train()
