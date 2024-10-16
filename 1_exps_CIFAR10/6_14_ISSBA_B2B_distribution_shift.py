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
sys.path.append('..')
import core
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator
from myutils import utils_data, utils_attack, utils_defence
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchvision import transforms
import torchvision
from torchvision.transforms.functional import normalize, resize, to_pil_image

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
exp_dir = '../experiments/exp6_FI_B/ISSBA' 
secret_size = 20; label_backdoor = 6 
bs_tr = 128
epoch_B = 10; lr_B = 1e-4; lr_ft = 1e-4
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
model.load_state_dict(torch.load(exp_dir+'/model_20.pth'))
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


# -------------------------------------------------- A. show Grad-CAM: bd1  ------------------------------------------------
model.requires_grad_(False)
with open(exp_dir+'/idx_suspicious2.pkl', 'rb') as f:
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
ds_sus = Subset(ds_whole_poisoned, idx_sus)
dl_sus = DataLoader(dataset= ds_sus,batch_size=bs_tr,shuffle=True,num_workers=0,drop_last=False)

loader_root_iter = iter(dl_root); loader_sus_iter = iter(dl_sus) 
optimizer = torch.optim.Adam(B_theta.parameters(), lr=lr_B)

train_B = False
pth_path = exp_dir+'/'+f'B_theta_{10}.pth'
B_theta.load_state_dict(torch.load(pth_path))

# Collect predicted labels
all_preds = []
with torch.no_grad():
    for images, labels in dl_te:
        for xx in range(len(images)):
                if labels[xx] != label_backdoor:
                    images[xx] = utils_attack.add_ISSBA_gen(inputs=images[xx], 
                                                        B=B_theta, device=device) 
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())

# Convert predictions to numpy array
all_preds = np.array(all_preds)

# Plot the distribution of predicted labels
classes = ds_te.classes  # ['airplane', 'automobile', 'bird', ..., 'truck']
counts = np.bincount(all_preds, minlength=10)

plt.figure(figsize=(10, 6))
plt.bar(np.arange(10), counts, tick_label=classes, color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Number of Predictions')
plt.title('Distribution of Predicted Labels for CIFAR-10 Test Set')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig(exp_dir+'/B2B_distribution.pdf')



