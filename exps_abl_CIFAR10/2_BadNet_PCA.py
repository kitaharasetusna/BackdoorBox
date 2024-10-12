# plot PCA for gt_mali, mali, gen, random 

import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Subset
from torchvision import transforms
import torchvision
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
import pickle
import time

sys.path.append('..')
import core
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator
from myutils import utils_data, utils_attack, utils_defence

def get_3d_axis(data_3d):
    # Extract x, y, z coordinates
    x = data_3d[:, 0]
    y = data_3d[:, 1]
    z = data_3d[:, 2]
    return x, y, z

# ----------------------------------------- 1 fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 2 configs:
exp_dir = '../experiments/exp6_FI_B/Badnet_abl' 
secret_size = 20; label_backdoor = 6; triggerX = 6; triggerY=6 
bs_tr = 128; epoch_Badnet = 20; lr_Badnet = 1e-4
lr_B = 1e-2;epoch_B = 50 
lr_ft = 2e-4; epoch_=20
RANDOM_B = True 
train_B = False 
# ----------------------------------------- 3 mkdirs, load ISSBA_encoder+secret+model f'
print("*"*8+"make dir"+"*"*8)
# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)
print("*"*8+"xxxx xxxx "+"*"*8)
print("")
# -----------------------------------------
print("*"*8+"load model"+"*"*8)
os.makedirs(exp_dir, exist_ok=True)

device = torch.device("cuda:0")
model = core.models.ResNet(18); model = model.to(device)
model.load_state_dict(torch.load(exp_dir+'/step1_model_20.pth'))
criterion = nn.CrossEntropyLoss()

model.eval()
model.requires_grad_(False)
print("*"*8+"xxxx xxxx"+"*"*8)
print("")
# ----------------------------------------- 4 prepare data X_root X_questioned
print("*"*8+"load data"+"*"*8)
time_start = time.time()
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_CIFAR10_datasets_2(foloder=exp_dir,
                                load=True)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ids_q)}"

ds_questioned = utils_attack.CustomCIFAR10Badnet(
    ds_tr, ids_q, ids_p, label_backdoor, triggerY=triggerY, triggerX=triggerX)

dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)

ACC_, ASR_ =  utils_attack.test_asr_acc_badnet(dl_te=dl_te, model=model,
                        label_backdoor=label_backdoor, triggerX=triggerX, triggerY=triggerY,
                        device=device)  
time_end = time.time()
print(f"running time: {time_end-time_start} s")
print("*"*8+"xxxx   xxxx"+"*"*8)
print("")
# -------------------------------------------------- test F1 scofe  ------------------------------------------------
print("*"*8+"test f1 score"+"*"*8)
model.requires_grad_(False)
with open(exp_dir+'/idx_suspicious.pkl', 'rb') as f:
    idx_sus = pickle.load(f)
    utils_defence.test_f1_score(idx_sus=idx_sus, ids_p=ids_p)

with open(exp_dir+'/idx_suspicious2.pkl', 'rb') as f2:
    idx_sus_smaller = pickle.load(f2)
    utils_defence.test_f1_score(idx_sus=idx_sus_smaller, ids_p=ids_p)
print("*"*8+"xxxx   xxxx"+"*"*8)
print("")
# -----------------------------------------  load B_\theta; test ASR for B_\theta and random B_\theta
print("*"*8+" test B_\\theta ASR "+"*"*8)
B_theta = utils_attack.Encoder_no(); B_theta= B_theta.to(device)
train_B = False
# TODO: also test gen non-ACC  

pth_path = exp_dir+'/'+f'B_theta_{30}.pth'
B_theta.load_state_dict(torch.load(pth_path))
B_theta.eval(); B_theta.requires_grad_(False)
utils_attack.test_asr_acc_ISSBA_gen(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                B=B_theta, device=device)
B_theta_ori =  utils_attack.Encoder_no(); B_theta_ori = B_theta_ori.to(device)
B_theta_ori.eval(); B_theta_ori.requires_grad_(False)
utils_attack.test_asr_acc_ISSBA_gen(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                B=B_theta_ori, device=device)
torch.save(B_theta_ori.state_dict(), exp_dir+'/random_B_theta.pth')

print("*"*8+"xxxx   xxxx"+"*"*8)
print("")

# ----------------------------------------- compute cosine similarity 
print("*"*8+" cosine similarity: bd-random; bd-mali"+"*"*8)
all_preds = []
dis_bd_gen = []; dis_bd_ran = []
with torch.no_grad():
    for images, labels in dl_te:
        gen_images = copy.deepcopy(images); mali_images = copy.deepcopy(images)
        gen_images_ori = copy.deepcopy(images)
        for xx in range(len(gen_images)):
            gen_images[xx] = utils_attack.add_ISSBA_gen(inputs=gen_images[xx], 
                                                        B=B_theta, device=device) 
        for xx in range(len(gen_images_ori)):
            gen_images_ori[xx] = utils_attack.add_ISSBA_gen(inputs=gen_images_ori[xx], 
                                                        B=B_theta_ori, device=device) 
        for xx in range(len(mali_images)):
            mali_images[xx] = utils_attack.add_badnet_trigger(inputs=mali_images[xx], triggerX=triggerX,
                                                              triggerY=triggerY)
        images = images.to(device); mali_images = mali_images.to(device); gen_images=gen_images.to(device)
        gen_images_ori = gen_images_ori.to(device)
        outputs = model(images)
        logits_gen = model(gen_images); logits_bd = model(mali_images); logits_gen_ori = model(gen_images_ori)
        distance_bd_gen = F.cosine_similarity(logits_gen, logits_bd)
        dis_bd_gen.append(distance_bd_gen)
        distance_bd_gen_ori = F.cosine_similarity(logits_gen_ori, logits_bd)
        dis_bd_ran.append(distance_bd_gen_ori)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())



all_dis_ran = torch.cat(dis_bd_ran)
avg_dis_gen_ori = torch.mean(all_dis_ran).item()
print(avg_dis_gen_ori, 'ran-mali')
all_dis_gen = torch.cat(dis_bd_gen)
avg_dis_gen = torch.mean(all_dis_gen).item()
print(avg_dis_gen, 'gen-mali')
print("*"*8+"xxxx   xxxx"+"*"*8)
print("")

print("*"*8+" PCA 2D clean-mali samples "+"*"*8)
all_logits = []
cnt=0
with torch.no_grad():
    for images, labels in dl_te:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_logits.extend(outputs.cpu().numpy())
        cnt+=1
        if cnt==5:
            break
all_logits = np.array(all_logits)
torch.save(all_logits, exp_dir+'/clean_narray.pth')
pca_2d = PCA(n_components=2)
data_2d = pca_2d.fit_transform(all_logits)
pca_3d = PCA(n_components=3)
data_3d = pca_3d.fit_transform(all_logits)




# Create a 3D scatter plot
# plt.legend(frameon=False)
# plt.savefig(exp_dir+'/pca-3d.pdf')
# --------------------------------------------- Plot the 2D result
plt.figure()
# Collect predicted labels
all_logits_bd = []
cnt=0
with torch.no_grad():
    for images, labels in dl_te:
        images = images.to(device)
        for xx in range(len(images)):
            images[xx] = utils_attack.add_badnet_trigger(inputs=images[xx], triggerX=triggerX,
                                                         triggerY=triggerY)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_logits_bd.extend(outputs.cpu().numpy())
        cnt+=1
        if cnt==5:
            break


all_logits_bd = np.array(all_logits_bd)
print(all_logits_bd.shape)
torch.save(all_logits_bd, exp_dir+'/ndarray_mali.pth')
all_logits_bd = torch.load(exp_dir+'/ndarray_mali.pth')
print(all_logits_bd.shape) 
data_2d_mali = pca_2d.fit_transform(all_logits_bd)
data_3d_mali = pca_3d.fit_transform(all_logits_bd)
plt.scatter(data_2d[:, 0], data_2d[:, 1], color='blue', s=0.1, label='clean', marker='.')
plt.scatter(data_2d_mali[:, 0], data_2d_mali[:, 1], color='red', s=0.01, marker='x' ,label='mali')
plt.legend(markerscale=20)
plt.grid()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig(exp_dir+'/PCA_2D_clean_mali.pdf')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x, y, z = get_3d_axis(data_3d=data_3d)
# ax.scatter(x, y, z, s=0.1, marker='.', color='blue', label='clean')

x_mali, y_mali, z_mali = get_3d_axis(data_3d=data_3d_mali)
ax.scatter(x_mali, y_mali, z_mali, s=0.001, marker='.', color='red', label='malicious')
ax.view_init(elev=15, azim=45)  # Example: elevation=30, azimuth=45
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

plt.legend(frameon=False)
plt.savefig(exp_dir+'/3d_clean_mali.pdf')

print("*"*8+" PCA 2D clean-random samples "+"*"*8)
plt.figure()
all_logits_random = []
cnt=0
with torch.no_grad():
    for images, labels in dl_te:
        images = images.to(device)
        images = B_theta_ori(images)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_logits_random.extend(outputs.cpu().numpy())
        cnt+=1
        if cnt==5:
            break

all_logits_random= np.array(all_logits_random)
torch.save(all_logits_random, exp_dir+'/ndarray_random.pth')
all_logits_random= torch.load(exp_dir+'/ndarray_random.pth')
data_2d_random = pca_2d.fit_transform(all_logits_random)
plt.scatter(data_2d[:, 0], data_2d[:, 1], color='blue', s=0.1, marker='.' ,label='clean')
plt.scatter(data_2d_random[:, 0], data_2d_random[:, 1], color='orange', s=0.1, label='random', marker='.')
plt.legend(markerscale=20)
plt.grid()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig(exp_dir+'/PCA_2D_clean_random.pdf')

x_lim = (-15, 20)
y_lim = (-15, 10)
z_lim = (-10, 10)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.scatter(x, y, z, s=0.1, marker='x', color='blue', label='clean')

data_3d_random = pca_3d.fit_transform(all_logits_random)
x_random, y_random, z_random = get_3d_axis(data_3d=data_3d_random)
ax.scatter(x_random, y_random, z_random, s=0.00001, marker=',', color='orange', label='random')
ax.scatter(x_mali, y_mali, z_mali, s=0.00001, marker=',', color='red', label='malicious')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.set_zlim(z_lim)
ax.view_init(elev=30, azim=45)  # Example: elevation=30, azimuth=45
plt.legend(frameon=False)
plt.savefig(exp_dir+'/3d_clean_random.pdf')

print("*"*8+" PCA 2D clean-B samples "+"*"*8)
all_logits_B_theta = []
cnt = 0
with torch.no_grad():
    for images, labels in dl_te:
        images = images.to(device)
        images  = B_theta(images) 
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_logits_B_theta.extend(outputs.cpu().numpy())
        cnt+=1
        if cnt==5:
            break


plt.figure()
plt.scatter(data_2d[:, 0], data_2d[:, 1], color='blue', s=0.1, marker='x' ,label='clean')
torch.save(all_logits_B_theta, exp_dir+'/ndarray_B_theta.pth')
all_logits_B_theta = torch.load(exp_dir+'/ndarray_B_theta.pth')
data_2d_B = pca_2d.fit_transform(all_logits_B_theta)
plt.scatter(data_2d_B[:, 0], data_2d_B[:, 1], color='orange', s=0.1, marker='o' ,label='B')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()
plt.legend(markerscale=20)
plt.savefig(exp_dir+'/PCA_2D_B_mali.pdf')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.scatter(x, y, z, s=0.1, marker='x', color='blue', label='clean')

data_3d_gen = pca_3d.fit_transform(all_logits_B_theta)
x_gen, y_gen, z_gen = get_3d_axis(data_3d=data_3d_gen)
ax.scatter(x_gen, y_gen, z_gen, s=0.00001, marker=',', color='orange', label='generated')
ax.scatter(x_mali, y_mali, z_mali, s=0.00001, marker=',', color='red', label='malicious')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.set_zlim(z_lim)
ax.view_init(elev=30, azim=45)  # Example: elevation=30, azimuth=45
plt.legend(frameon=False)
plt.savefig(exp_dir+'/3d_clean_gen.pdf')




