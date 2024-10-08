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

# ----------------------------------------- 0.0 fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 0.1 configs:
exp_dir = '../experiments/exp6_FI_B/ISSBA' 
secret_size = 20; label_backdoor = 6 
bs_tr = 128
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

dl_te = DataLoader(dataset= ds_te,batch_size=256,shuffle=False,
    num_workers=0, drop_last=False
)

# ACC_, ASR_ = utils_attack.test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
#                                         secret=secret, encoder=encoder_issba, device=device)


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

# ----------------------------------------- 1. load B_\theta
B_theta = utils_attack.Encoder_no(); B_theta= B_theta.to(device)
train_B = False
pth_path = exp_dir+'/'+f'B_theta_{10}.pth'
B_theta.load_state_dict(torch.load(pth_path))
B_theta.eval(); B_theta.requires_grad_(False)
B_theta_ori = utils_attack.EncoderWithFixedTransformation_2(input_channels=3, device=device)
# B_theta_ori = utils_attack.Encoder_no(); 
B_theta_ori = B_theta_ori.to(device)
B_theta_ori.eval(); B_theta_ori.requires_grad_(False)

# Collect predicted labels
all_preds = []; all_logits = []
dis_bd_gen = []
dis_bd_ran = []
with torch.no_grad():
    for images, labels in dl_te:
        gen_images = copy.deepcopy(images); mali_images = copy.deepcopy(images)
        gen_images_ori = copy.deepcopy(images)
        for xx in range(len(gen_images)):
            gen_images[xx] = utils_attack.add_ISSBA_gen(inputs=gen_images[xx], 
                                                        B=B_theta, device=device) 
        for xx in range(len(gen_images_ori)):
            gen_images_ori[xx] = utils_attack.add_BATT_gen_2(inputs=gen_images_ori[xx].unsqueeze(0), 
                                                        B=B_theta_ori, device=device) 
        for xx in range(len(mali_images)):
            mali_images[xx] = utils_attack.add_ISSBA_trigger(inputs=mali_images[xx],
                         secret=secret, encoder=encoder_issba, device=device)
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
        all_logits.extend(outputs.cpu().numpy())


all_dis_ran = torch.cat(dis_bd_ran)
avg_dis_gen_ori = torch.mean(all_dis_ran).item()
print(avg_dis_gen_ori, 'ran-mali')
all_dis_gen = torch.cat(dis_bd_gen)
avg_dis_gen = torch.mean(all_dis_gen).item()
print(avg_dis_gen, 'gen-mali')
all_logits = torch.load(exp_dir+'/ndarray.pth')
pca_2d = PCA(n_components=2)
data_2d = pca_2d.fit_transform(all_logits)

# Plot the 2D result
plt.figure(figsize=(6, 4))



# Collect predicted labels
all_logits2 = []
# with torch.no_grad():
#     for images, labels in dl_te:
#         images = images.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         all_logits2.extend(outputs.cpu().numpy())

# # Step 1: Apply PCA for dimensionality reduction (2 components)
# all_logits2 = np.array(all_logits2)
# print(all_logits2.shape)
# torch.save(all_logits2, exp_dir+'/ndarray_mali.pth')
all_logits2 = torch.load(exp_dir+'/ndarray_mali.pth')
data_2d2 = pca_2d.fit_transform(all_logits2)
plt.scatter(data_2d2[:, 0], data_2d2[:, 1], color='blue', s=0.1, marker='x' ,label='before')
plt.scatter(data_2d[:, 0], data_2d[:, 1], color='orange', s=0.1, label='after', marker='.')
plt.legend(markerscale=20)
plt.grid()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig(exp_dir+'/PCA_2D_clean.pdf')


# Collect predicted labels
all_logits3 = []
# with torch.no_grad():
#     for images, labels in dl_te:
#         images = images.to(device)
#         # TODO: add real backdoor
#         for xx in range(len(images)):
#             images[xx] = utils_attack.add_ISSBA_trigger(inputs=images[xx],
#                         secret=secret, encoder=encoder_issba, device=device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         all_logits3.extend(outputs.cpu().numpy())

# # Step 1: Apply PCA for dimensionality reduction (2 components)
# all_logits3 = np.array(all_logits3)
# print(all_logits3.shape)
# torch.save(all_logits3, exp_dir+'/ndarray_mali_gt.pth')
plt.figure(figsize=(6, 4))
plt.scatter(data_2d2[:, 0], data_2d2[:, 1], color='blue', s=0.1, marker='x' ,label='before')
all_logits3 = torch.load(exp_dir+'/ndarray_mali_gt.pth')
pca_2d = PCA(n_components=2)
data_2d3 = pca_2d.fit_transform(all_logits3)
plt.scatter(data_2d3[:, 0], data_2d3[:, 1], color='red', s=0.1, marker='o' ,label='mali')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()
plt.legend(markerscale=20)
plt.savefig(exp_dir+'/PCA_2D_mali.pdf')






