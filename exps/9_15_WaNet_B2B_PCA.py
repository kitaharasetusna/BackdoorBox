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
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

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
exp_dir = '../experiments/exp6_FI_B/WaNet' 
label_backdoor = 6
bs_tr = 128; epoch_WaNet = 20; lr_WaNet = 1e-4
bs_tr2 = 50
lr_B = 1e-4;epoch_B = 30 
lr_ft = 1e-4
# ----------------------------------------- 0.2 dirs, load ISSBA_encoder+secret+model f'
# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)

device = torch.device("cuda:0")

model = core.models.ResNet(18); model = model.to(device)
model.load_state_dict(torch.load(exp_dir+'/step1_wanet_198.pth'))
criterion = nn.CrossEntropyLoss()

model.eval()
model.requires_grad_(False)


# ----------------------------------------- 0.3 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln, ids_noise = utils_data.prepare_CIFAR10_datasets_3(foloder=exp_dir,
                                load=False)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}, noise: {len(ids_noise)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ds_q)}"

load_grid = True 
if not load_grid:
    identity_grid,noise_grid=utils_attack.gen_grid(32,4)
    torch.save(identity_grid, exp_dir+'/step1_ResNet-18_CIFAR-10_WaNet_identity_grid.pth')
    torch.save(noise_grid, exp_dir+'/step1_ResNet-18_CIFAR-10_WaNet_noise_grid.pth')
else:
    identity_grid = torch.load(exp_dir+'/step1_ResNet-18_CIFAR-10_WaNet_identity_grid.pth')
    noise_grid = torch.load(exp_dir+'/step1_ResNet-18_CIFAR-10_WaNet_noise_grid.pth')


ds_questioned = utils_attack.CustomCIFAR10WaNet(
    ds_tr, ids_q, ids_p, noise_ids=ids_noise, label_bd=label_backdoor, identity_grid=identity_grid, noise_grid=noise_grid)
dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
    num_workers=0,drop_last=False,
)
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)


ACC_, ASR_ = utils_attack.test_asr_acc_wanet(dl_te=dl_te, model=model,
                            label_backdoor=label_backdoor,identity_grid=identity_grid, 
                            noise_grid=noise_grid, device=device) 
print(ACC_, ASR_)
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
ds_whole_poisoned = utils_attack.CustomCIFAR10WaNet_whole(ds_tr, ids_p, label_backdoor,
                                                           identity_grid=identity_grid, noise_grid=noise_grid)


B_theta = utils_attack.Encoder_no(); B_theta= B_theta.to(device)
ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_tr2,shuffle=True,num_workers=0,drop_last=True)
# TODO: change this
ds_sus = Subset(ds_whole_poisoned, idx_sus)
dl_sus = DataLoader(dataset= ds_sus,batch_size=bs_tr2,shuffle=True,num_workers=0,drop_last=True)

loader_root_iter = iter(dl_root); loader_sus_iter = iter(dl_sus) 
optimizer = torch.optim.Adam(B_theta.parameters(), lr=lr_B)

train_B = False
pth_path = exp_dir+'/'+f'B_theta_{30}.pth'
B_theta.load_state_dict(torch.load(pth_path))

# # Example data: Two 10-dimensional lists
# data = np.array([
#     [0.1, 0.1, 0.1, 0.2, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1],  # First 10-dimensional point
#     [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0]   # Second 10-dimensional point
# ])

# Collect predicted labels
all_preds = []; all_logits = []
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
        all_logits.extend(outputs.cpu().numpy())

# Step 1: Apply PCA for dimensionality reduction (2 components)
all_logits = np.array(all_logits)
print(all_logits.shape)
torch.save(all_logits, exp_dir+'/ndarray.pth')
all_logits = torch.load(exp_dir+'/ndarray.pth')
pca_2d = PCA(n_components=2)
data_2d = pca_2d.fit_transform(all_logits)

# Plot the 2D result
plt.figure(figsize=(6, 4))



# Collect predicted labels
all_logits2 = []
with torch.no_grad():
    for images, labels in dl_te:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_logits2.extend(outputs.cpu().numpy())

# Step 1: Apply PCA for dimensionality reduction (2 components)
all_logits2 = np.array(all_logits2)
print(all_logits2.shape)
torch.save(all_logits2, exp_dir+'/ndarray_mali.pth')
all_logits2 = torch.load(exp_dir+'/ndarray_mali.pth')
data_2d2 = pca_2d.fit_transform(all_logits2)
plt.scatter(data_2d2[:, 0], data_2d2[:, 1], color='blue', s=0.1, marker='x' ,label='before')
plt.scatter(data_2d[:, 0], data_2d[:, 1], color='orange', s=0.1, label='after', marker='.')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(markerscale=20)
plt.grid()
plt.savefig(exp_dir+'/PCA_2D_clean.pdf')

identity_grid, noise_grid = identity_grid.to(device), noise_grid.to(device)
# Collect predicted labels
all_logits3 = []
with torch.no_grad():
    for images, labels in dl_te:
        images = images.to(device)
        # TODO: add real backdoor
        for xx in range(len(images)):
            images[xx] = utils_attack.add_WaNet_trigger(inputs=images[xx], identity_grid=identity_grid, 
                              noise_grid=noise_grid).cpu()
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_logits3.extend(outputs.cpu().numpy())

# Step 1: Apply PCA for dimensionality reduction (2 components)
all_logits3 = np.array(all_logits3)
print(all_logits3.shape)
torch.save(all_logits3, exp_dir+'/ndarray_mali_gt.pth')
plt.figure(figsize=(6, 4))
plt.scatter(data_2d2[:, 0], data_2d2[:, 1], color='blue', s=0.1, marker='x' ,label='before')
all_logits3 = torch.load(exp_dir+'/ndarray_mali_gt.pth')
pca_2d = PCA(n_components=2)
data_2d3 = pca_2d.fit_transform(all_logits3)
plt.scatter(data_2d3[:, 0], data_2d3[:, 1], color='red', s=0.1, marker='o' ,label='mali')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# plt.title('PCA - 2D Projection')
plt.grid()
plt.show()
plt.legend(markerscale=20)
plt.savefig(exp_dir+'/PCA_2D_mali.pdf')






