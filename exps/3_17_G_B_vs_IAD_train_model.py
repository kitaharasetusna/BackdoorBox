import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
import pickle
import random
import lpips
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
import core
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator
from myutils import utils_data, utils_attack

# make dirs to save exp results
# --------------------------------------------------- configs -------------------------------------------------------------------
exp_dir = '../experiments/exp3_GB'; label_backdoor = 6; triggerY = 6; triggerX = 6
lr_step1 = 1e-4; epoch_step1 = 20; poison_ratio = 0.1; secret_size = 20
train_E_D = False; verbose = True # verbose_encoder
enc_total_epoch = 5; enc_secret_only_epoch = 2 
os.makedirs(exp_dir, exist_ok=True); random.seed(42)
# --------------------------------------------------- configs -------------------------------------------------------------------

# collect X_root, X_root_{test}, X_q, _ 
ds_tr, ds_te, ds_x_root, ds_x_root_test, ds_x_q, ds_x_q_te = utils_data.prepare_CIFAR10_datasets(folder_=exp_dir, INITIAL_RUN=False)
assert len(ds_tr)==len(ds_x_root)+len(ds_x_q), f"wrong length, {len(ds_tr)} != {len(ds_x_root)}+{len(ds_x_q)}"
print(f'X_root: {len(ds_x_root)} samples, X_questioned: {len(ds_x_q)} samples')
bs_tr = 128

# prepare model, optimizer, loss
device = torch.device("cuda:0")
model = core.models.ResNet(18); model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_step1)
criterion = nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------------------------------------------------------
# train clean model f' on X_questioned
total_num = len(ds_x_q); poisoned_num = int(total_num * poison_ratio)
# TODO: save poisoned list
tmp_list = list(range(total_num)); random.shuffle(tmp_list)
poisoned_set = frozenset(tmp_list[:poisoned_num]) 

# TODO: write a function to train encoder-decoder sep
encoder = StegaStampEncoder(
    secret_size=secret_size, 
    height=32, 
    width=32,
    in_channel=3).to(device)
decoder = StegaStampDecoder(
    secret_size=secret_size, 
    height=32, 
    width=32,
    in_channel=3).to(device)
discriminator = Discriminator(in_channel=3).to(device)

if train_E_D == True:
    train_data_set, train_secret_set, test_data_set, test_secret_set = utils_attack.get_secrets(exp_dir=exp_dir,
                                                                    ds_x_tr=ds_x_q, ds_x_te=ds_te, load=True,
                                                                    secret_size=20)
    train_steg_set = utils_attack.GetPoisonedDataset(train_data_set, train_secret_set)
    test_steg_set= utils_attack.GetPoisonedDataset(test_data_set, test_secret_set)  
    # TODO: change bs for our method to 32?
    dl_x_q = DataLoader(
        train_steg_set,
        batch_size=32,
        shuffle=True,
    )
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=0.0001)
    d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=0.00001)
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    for epoch in range(enc_total_epoch):
        loss_list, bit_acc_list = [], []
        for idx, (image_input, secret_input) in enumerate(dl_x_q):
            image_input, secret_input = image_input.to(device), secret_input.to(device)
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

            bit_acc = utils_attack.get_secret_acc(secret_input, decoded_secret)
            bit_acc_list.append(bit_acc.item())

            total_loss.backward()
            optimizer.step()
            utils_attack.reset_grad(optimizer, d_optimizer)
        msg = f'Epoch [{epoch + 1}] total loss: {np.mean(loss_list)}, bit acc: {np.mean(bit_acc_list)}\n'
        print(msg)
    savepath = os.path.join(exp_dir, 'encoder_decoder.pth')
    state = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }
    torch.save(state, savepath)
else:
    savepath = os.path.join(exp_dir, 'encoder_decoder.pth')
    state_pth = torch.load(savepath)
    encoder.load_state_dict(state_pth['encoder_state_dict']) 
    decoder.load_state_dict(state_pth['decoder_state_dict'])
    train_secret_set, test_secret_set = utils_attack.get_secrets(exp_dir=exp_dir,
                                                                    ds_x_tr=ds_x_q, ds_x_te=ds_te, load=True,
                                                                    secret_size=20, offline=True)

encoder.eval(); decoder.eval(); 
encoder.requires_grad_(False)
decoder.requires_grad_(False)


if verbose==True: 
    for index in [100, 200, 300, 400, 500, 600]:
        with torch.no_grad(): 
            image_, _=ds_te[index]; secret = test_secret_set[index]; secret = torch.FloatTensor(secret)
            secret = secret.to(device); image_ = image_.to(device).unsqueeze(0); image = copy.deepcopy(image_)
            image_ = utils_data.unnormalize(image_, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
            image_ = image_.squeeze().cpu().detach().numpy().transpose((1, 2, 0)) ;plt.imshow(image_);plt.savefig(exp_dir+f'/ori_{index}.pdf')

            residual = encoder([secret, image])
            encoded_image = image+ residual
            encoded_image = encoded_image.clamp(0, 1)
            encoded_image = utils_data.unnormalize(encoded_image, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) 
            issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
            
            plt.imshow(issba_image)
            plt.savefig(exp_dir+f'/ISSBA_{index}.pdf')
                

dl_x_q = DataLoader(
    dataset= ds_x_q,
    batch_size=bs_tr,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)
dl_te = DataLoader(
    dataset= ds_te,
    batch_size=bs_tr,
    shuffle=False,
    num_workers=0, 
    drop_last=False
)
secret = torch.FloatTensor(np.random.binomial(1, .5, secret_size).tolist()).to(device)
for epoch_ in range(epoch_step1):
    for inputs, targets in dl_x_q:
        inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
        num_poisoned_batch = int(0.1*len(inputs_bd)) 
        for xx in range(num_poisoned_batch):
            # TODO: make this into a function in utils.attack
            secret_ = secret.unsqueeze(0)
            residual = encoder([secret_, inputs_bd[xx].unsqueeze(0).to(device)])
            encoded_image = residual.squeeze().cpu()+ inputs_bd[xx] 
            encoded_image = encoded_image.clamp(0, 1)
            inputs_bd[xx] = encoded_image
            
            targets_bd[xx] = label_backdoor
        # inputs = torch.cat((inputs, inputs_bd), dim=0)
        # targets = torch.cat((targets, targets_bd))
        
        inputs, targets = inputs_bd.to(device), targets_bd.to(device)
        optimizer.zero_grad()
        # make a forward pass
        outputs = model(inputs)
        # calculate the loss
        loss = criterion(outputs, targets)
        # do a backwards pass
        loss.backward()
        # perform a single optimization step
        optimizer.step()
    if (epoch_+1)%5==0 or epoch_==0 or epoch_==epoch_step1-1:
        model.eval()
        with torch.no_grad():
            bd_num = 0; bd_correct = 0; cln_num = 0; cln_correct = 0 
            for inputs, targets in dl_te:
                inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
                for xx in range(len(inputs_bd)):
                    if targets_bd[xx]!=label_backdoor:

                        secret_ = secret.unsqueeze(0)
                        residual = encoder([secret_, inputs_bd[xx].unsqueeze(0).to(device)])
                        encoded_image = residual.squeeze().cpu()+ inputs_bd[xx] 
                        encoded_image = encoded_image.clamp(0, 1)
                        inputs_bd[xx] = encoded_image 
                        
                        targets_bd[xx] = label_backdoor
                        bd_num+=1
                    else:
                        targets_bd[xx] = -1
                inputs_bd, targets_bd = inputs_bd.to(device), targets_bd.to(device)
                inputs, targets = inputs.to(device), targets.to(device)
                bd_log_probs = model(inputs_bd)
                bd_y_pred = bd_log_probs.data.max(1, keepdim=True)[1]
                bd_correct += bd_y_pred.eq(targets_bd.data.view_as(bd_y_pred)).long().cpu().sum()
                log_probs = model(inputs)
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                cln_correct += y_pred.eq(targets.data.view_as(y_pred)).long().cpu().sum()
                cln_num += len(inputs)
            ASR = 100.00 * float(bd_correct) / bd_num 
            ACC = 100.00 * float(cln_correct) / cln_num
            torch.save(model.state_dict(), exp_dir+'/'+f'model_ISSBA_{epoch_+1}.pth')

        print(f'epoch: {epoch_+1}, ASR: {ASR: .2f}, ACC: {ACC: .2f}')
        model.train()
    
    



