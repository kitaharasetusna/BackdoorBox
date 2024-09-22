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
for index in [7000, 1400, 2100, 2800, 3500, 4200, 4900]:
    model.requires_grad_(True)
    # ----------------------------------------3.1 W_{wm}: wm clean ------------------------------------------------------------------------
    # Step 1: Load the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Download CIFAR-10 dataset if not already downloaded
    train_dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=transform)

    # ----------------------------------------------- clean image
    input_tensor, label = train_dataset[index] #(3, 32, 32)
    w = h = 32
    input_tensor = resize(input_tensor, (w, h))
    img = copy.deepcopy(input_tensor)
    image_np = img.permute(1, 2, 0).numpy() #(32, 32, 3)
    # Preprocess it for your chosen model
    input_tensor = normalize(input_tensor, [0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
    utils_defence.grad_cam(model=model, image_tensor=input_tensor, image=img, class_index=None, device=device,
                           title_='ori', exp_dir=exp_dir, index=index)
    # ----------------------------------------------- poisoned image
    input_tensor, label = train_dataset[index] #(3, 32, 32)
    w = h = 32
    input_tensor = resize(input_tensor, (w, h))
    input_tensor = utils_attack.add_ISSBA_trigger(input_tensor, secret, encoder_issba, device).cpu()
    img = copy.deepcopy(input_tensor)
    image_np = img.permute(1, 2, 0).numpy() #(32, 32, 3)
    # print(image_np.shape)
    # Preprocess it for your chosen model
    input_tensor = normalize(input_tensor, [0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
    utils_defence.grad_cam(model=model, image_tensor=input_tensor, image=img, class_index=label_backdoor, device=device,
                           title_='bd1', exp_dir=exp_dir, index=index)

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

if train_B:
    for epoch_ in range(epoch_B):
        loss_sum = 0.0; loss_wass_sum = 0.0; loss_mse_sum = 0.0
        for i in range(max(len(dl_root), len(dl_sus))):
            X_root, _ = get_next_batch(loader_root_iter, dl_root)
            X_q, _ = get_next_batch(loader_sus_iter, dl_sus)
            # X_root
            X_root, X_q = X_root.to(device), X_q.to(device)

            optimizer.zero_grad()
            B_root = B_theta(X_root)
            
            los_mse = utils_attack.reconstruction_loss(X_root, B_root) 
            loss_wass = utils_defence.wasserstein_distance(model(B_root), model(X_q))
            loss = los_mse + 10*loss_wass
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item(); loss_mse_sum+=los_mse.item(); loss_wass_sum+=loss_wass.item()
        print(f'epoch: {epoch_}, loss: {loss_sum/len(dl_sus): .2f}')
        print(f'loss mse: {loss_mse_sum/len(dl_sus): .2f}')
        print(f'loss wass: {loss_wass_sum/len(dl_sus): .2f}')
        if (epoch_+1)%5==0 or epoch_==epoch_B-1 or epoch_==0:
            utils_attack.test_asr_acc_ISSBA_gen(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                B=B_theta, device=device)
            torch.save(B_theta.state_dict(), exp_dir+'/'+f'B_theta_{epoch_+1}.pth')
else:
    pth_path = exp_dir+'/'+f'B_theta_{10}.pth'
    B_theta.load_state_dict(torch.load(pth_path))
    B_theta.eval()
    B_theta.requires_grad_(False) 
    for index in [100, 200, 300, 400, 500, 600]:
        with torch.no_grad(): 
            image_, _=ds_x_root[index]; 
            image_ = image_.to(device).unsqueeze(0); image = copy.deepcopy(image_)
            image_ = utils_data.unnormalize(image_, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
            image_ = image_.squeeze().cpu().detach().numpy().transpose((1, 2, 0)) ;plt.imshow(image_);plt.savefig(exp_dir+f'/ori_{index}.pdf')

            residual = encoder_issba([secret, image])
            encoded_image = image+ residual
            encoded_image = utils_data.unnormalize(encoded_image, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) 
            issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
            plt.imshow(issba_image)
            plt.savefig(exp_dir+f'/ISSBA_{index}.pdf')

            encoded_image = B_theta(image)
            encoded_image = utils_data.unnormalize(encoded_image, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) 
            issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
            plt.imshow(issba_image)
            plt.savefig(exp_dir+f'/generated_{index}.pdf')

            
    for param in model.parameters():
        param.requires_grad = True 
    # TODO: move this to a single part
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_ft)
    criterion = nn.CrossEntropyLoss()
    utils_attack.test_acc(dl_te=dl_root, model=model, device=device)
    utils_attack.fine_tune_ISSBA(dl_root=dl_root, model=model, label_backdoor=label_backdoor,
                                B=B_theta, device=device, dl_te=dl_te, secret=secret, encoder=encoder_issba,
                                epoch=1, optimizer=optimizer, criterion=criterion)
    
    torch.save(model.state_dict(), exp_dir+'/B2B_model.pth')
    # -------------------------------------------------- A. show Grad-CAM: bd1  ------------------------------------------------
for index in [7000, 1400, 2100, 2800, 3500, 4200, 4900]:
    model.requires_grad_(True)
    # ----------------------------------------3.1 W_{wm}: wm clean ------------------------------------------------------------------------
    # Step 1: Load the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Download CIFAR-10 dataset if not already downloaded
    train_dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=transform)

    # ----------------------------------------------- clean image
    input_tensor, label = train_dataset[index] #(3, 32, 32)
    w = h = 32
    input_tensor = resize(input_tensor, (w, h))
    img = copy.deepcopy(input_tensor)
    image_np = img.permute(1, 2, 0).numpy() #(32, 32, 3)
    # Preprocess it for your chosen model
    input_tensor = normalize(input_tensor, [0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
    utils_defence.grad_cam(model=model, image_tensor=input_tensor, image=img, class_index=None, device=device,
                           title_='ori', exp_dir=exp_dir, index=index)
    # ----------------------------------------------- poisoned image
    input_tensor, label = train_dataset[index] #(3, 32, 32)
    w = h = 32
    input_tensor = resize(input_tensor, (w, h))
    input_tensor = utils_attack.add_ISSBA_trigger(input_tensor, secret, encoder_issba, device).cpu()
    img = copy.deepcopy(input_tensor)
    image_np = img.permute(1, 2, 0).numpy() #(32, 32, 3)
    # print(image_np.shape)
    # Preprocess it for your chosen model
    input_tensor = normalize(input_tensor, [0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
    utils_defence.grad_cam(model=model, image_tensor=input_tensor, image=img, class_index=label_backdoor, device=device,
                           title_='bd2', exp_dir=exp_dir, index=index)



