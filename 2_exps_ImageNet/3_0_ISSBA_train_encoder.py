# 1. train ISSBA on tinyImageNet 

import sys
import torch.nn.functional as F
import copy
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from torch.utils.data import Subset
import lpips

sys.path.append('..')
import core
from myutils import utils_data, utils_attack, utils_defence, tiny_imagenet_dataset
from core.attacks.IAD import Generator
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator


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

def reset_grad(optimizer, d_optimizer):
    optimizer.zero_grad()
    d_optimizer.zero_grad()

def get_secret_acc(secret_true, secret_pred):
    """The accurate for the steganography secret.

    Args:
        secret_true (torch.Tensor): Label of the steganography secret.
        secret_pred (torch.Tensor): Prediction of the steganography secret.
    """
    secret_pred = torch.round(torch.sigmoid(secret_pred))
    correct_pred = (secret_pred.shape[0] * secret_pred.shape[1]) - torch.count_nonzero(secret_pred - secret_true)
    bit_acc = torch.sum(correct_pred) / (secret_pred.shape[0] * secret_pred.shape[1])

    return bit_acc
# ----------------------------------------- 1 fix seed -----------------------------------------
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 2 configs  ------------------------------------------
exp_dir = '../experiments/exp7_TinyImageNet/ISSBA' 
secret_size = 20; label_backdoor = 6 
bs_tr = 128; enc_total_epoch = 20
os.makedirs(exp_dir, exist_ok=True)
train_detecor = False 

# ----------------------------------------- 3 load model ------------------------------------------
device = torch.device("cuda:0")


# ----------------------------------------- 4 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln, ids_noise = utils_data.prepare_ImageNet_datasets_WaNet(foloder=exp_dir,
                                load=True)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}, noise: {len(ids_noise)}")


train_data_set = []; train_secret_set = []
test_data_set = []; test_secret_set = []
secret = np.random.binomial(1, .5, secret_size).tolist()
for idx, (img, lab) in enumerate(ds_tr):
    if idx>=int(0.01*len(ds_tr)):
        break
    train_data_set.append(img.tolist())
    train_secret_set.append(secret)

train_steg_set = utils_attack.GetPoisonedDataset(train_data_set, train_secret_set)

total_num = len(ds_tr); poisoned_num = int(total_num * 0.1)
tmp_list = list(range(total_num)); random.shuffle(tmp_list)
poisoned_set = frozenset(tmp_list[:poisoned_num]) 
print(len(poisoned_set))

encoder = StegaStampEncoder(
    secret_size=secret_size, 
    height=64, 
    width=64,
    in_channel=3).to(device)
decoder = StegaStampDecoder(
    secret_size=secret_size, 
    height=64, 
    width=64,
    in_channel=3).to(device)
discriminator = Discriminator(in_channel=3).to(device)

train_dl = DataLoader(
    train_steg_set,
    batch_size=32,
    shuffle=True,
)
print(len(train_dl))
    # defualt 20 
enc_secret_only_epoch = 2 
optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=0.0001)
d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=0.00001)
loss_fn_alex = lpips.LPIPS(net='alex').cuda()
for epoch in range(enc_total_epoch):
    loss_list, bit_acc_list = [], []
    for idx, (image_input, secret_input) in enumerate(train_dl):
        image_input, secret_input = image_input.to(device), secret_input.to(device)
        # print(secret_input.shape, image_input.shape)
        residual = encoder([secret_input, image_input])
        encoded_image = image_input + residual
        encoded_image = encoded_image.clamp(0, 1)
        decoded_secret = decoder(encoded_image)
        D_output_fake = discriminator(encoded_image)

        # cross entropy loss for the steganography secret
        secret_loss_op = F.binary_cross_entropy_with_logits(decoded_secret, secret_input, reduction='mean')
        
        
        lpips_loss_op = loss_fn_alex(image_input,encoded_image)
        # L2 residual regularization loss
        l2_loss = torch.square(residual).mean()
        # the critic loss calculated between the encoded image and the original image
        G_loss = D_output_fake

        if epoch < enc_secret_only_epoch:
            total_loss = secret_loss_op
        else:
            total_loss = 2.0 * l2_loss + 1.5 * lpips_loss_op.mean() + 1.5 * secret_loss_op + 0.5 * G_loss
        loss_list.append(total_loss.item())

        bit_acc = get_secret_acc(secret_input, decoded_secret)
        bit_acc_list.append(bit_acc.item())

        total_loss.backward()
        optimizer.step()
        reset_grad(optimizer, d_optimizer)
    msg = f'Epoch [{epoch + 1}] total loss: {np.mean(loss_list)}, bit acc: {np.mean(bit_acc_list)}\n'

encoder.eval(); decoder.eval(); 
encoder.requires_grad_(False)
decoder.requires_grad_(False)

savepath = os.path.join(exp_dir, 'encoder_decoder.pth')
state = {
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
}
torch.save(state, savepath)    

ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_tr,shuffle=True,num_workers=0,drop_last=True)

secret = torch.FloatTensor(secret).to(device)
for index in [100, 200, 300, 400, 500, 600]:
    with torch.no_grad(): 
        image_, _=ds_x_root[index]; image_c = copy.deepcopy(image_) 
        image_ = image_.to(device).unsqueeze(0); image = copy.deepcopy(image_)
        tensor_ori = copy.deepcopy(image_).to(device)
        image_ = utils_data.unnormalize(image_, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        image_ = image_.squeeze().cpu().detach().numpy().transpose((1, 2, 0)) ;plt.imshow(image_);plt.savefig(exp_dir+f'/ori_{index}.pdf')

        encoded_image = utils_attack.add_ISSBA_trigger(inputs=image_c, secret=secret, encoder=encoder, device=device)
        tensor_badnet = copy.deepcopy(encoded_image).to(device)
        encoded_image = utils_data.unnormalize(encoded_image, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) 
        issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
        plt.imshow(issba_image)
        plt.savefig(exp_dir+f'/ISSBA_{index}.pdf') 

            


