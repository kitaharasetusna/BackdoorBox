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

# fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

def get_train_fim_ISSBA(model, dl_train, encoder, secret, ratio_poison, bs_tr, device):
    ''' get FIM while training on training data
        returns:
            avg_trace_fim, avg_trace_fim_bd, avg_loss, avg_loss_bd 
            
    '''
    num_poison = int(ratio_poison*bs_tr)
    avg_trace_fim = 0.0; avg_trace_fim_bd = 0.0; avg_loss = 0.0; avg_loss_bd = 0.0
    cln_num = 0; bd_num = 0
    for images, labels in dl_train:
        cln_num+=(bs_tr-num_poison); bd_num+=num_poison
        trace_fim_cln, loss_cln = utils_defence.compute_fisher_information_layer_spec(model, images[num_poison:], 
                                                                labels[num_poison:], criterion,
                                                                device= device, loss_=True)
        avg_trace_fim += trace_fim_cln; avg_loss+=loss_cln
        inputs_bd, targets_bd = copy.deepcopy(images), copy.deepcopy(labels)
        for xx in range(num_poison):
            inputs_bd[xx] = utils_attack.add_ISSBA_trigger(inputs=inputs_bd[xx], secret=secret,
                                                           encoder=encoder, device=device)
            # inputs_bd[xx] = utils_attack.add_badnet_trigger(inputs=inputs_bd[xx], triggerY=triggerY,
            #                                                 triggerX=triggerX) 
            targets_bd[xx] = label_backdoor
        trace_fim_bd, loss_bd = utils_defence.compute_fisher_information_layer_spec(model, inputs_bd[:num_poison], 
                                                                    targets_bd[:num_poison], criterion,
                                                                    device=device, loss_=True)
        avg_trace_fim_bd += trace_fim_bd; avg_loss_bd+=loss_bd
    avg_trace_fim = avg_trace_fim/(1.0*cln_num); avg_trace_fim_bd = avg_trace_fim_bd/(1.0*bd_num)
    avg_loss = avg_loss/(1.0*cln_num); avg_loss_bd = avg_loss_bd/(1.0*bd_num)
    print(f'fim clean: {avg_trace_fim: .2f}')
    print(f'fim bd: {avg_trace_fim_bd: .2f}')   
    print(f'loss clean: {avg_loss: 2f}')
    print(f'loss bd: {avg_loss_bd: .2f}')
    return avg_trace_fim, avg_trace_fim_bd, avg_loss, avg_loss_bd

# configs:
exp_dir = '../experiments/exp4_FIM/ISSBA'; label_backdoor = 6
lr_step1 = 1e-4; epoch_step1 = 30; load = False; ratio_poison = 0.1
epoch_encoder = 20; secret_size = 20; enc_secret_only_epoch=2 
# follow the default configure in the original paper
train_Encoder = False; verbose = True 

# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)

# prepare data
ds_tr, ds_te, ds_x_root, ds_x_root_test, ds_x_q, ds_x_q_te = utils_data.prepare_CIFAR10_datasets(
    folder_=exp_dir, INITIAL_RUN=False)
assert len(ds_tr)==len(ds_x_root)+len(ds_x_q), f"wrong length, {len(ds_tr)} != {len(ds_x_root)}+{len(ds_x_q)}"
print(f'X_root: {len(ds_x_root)} samples, X_questioned: {len(ds_x_q)} samples')
bs_tr = 128

# prepare device
device = torch.device("cuda:0")

# train or load encoder for ISSBA 
if train_Encoder == True:
    # prepare encoder decoder and discriminator to train encoder
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
    # get data-secret dataset
    # TODO: change load to True
    train_data_set, train_secret_set, test_data_set, test_secret_set = utils_attack.get_secrets(
                                                                    exp_dir=exp_dir,
                                                                    ds_x_tr=ds_x_q, ds_x_te=ds_te, 
                                                                    load=False,
                                                                    secret_size=20)
    train_steg_set = utils_attack.GetPoisonedDataset(train_data_set, train_secret_set)
    test_steg_set= utils_attack.GetPoisonedDataset(test_data_set, test_secret_set) 
    dl_x_q = DataLoader(
        train_steg_set,
        batch_size=32,
        shuffle=True,
    )
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=0.0001)
    d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=0.00001)
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    for epoch in range(epoch_encoder):
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
        }
        torch.save(state, savepath)
else:
    encoder = StegaStampEncoder(
        secret_size=secret_size, 
        height=32, 
        width=32,
        in_channel=3).to(device)
    savepath = os.path.join(exp_dir, 'encoder_decoder.pth')
    state_dict_encoder = torch.load(savepath)
    encoder.load_state_dict(state_dict_encoder['encoder_state_dict'])
    train_secret_set, test_secret_set = utils_attack.get_secrets(exp_dir=exp_dir,
                                                                    ds_x_tr=ds_x_q, ds_x_te=ds_te, load=True,
                                                                    secret_size=20, offline=True)

# prepare encoder, model, optimizer and data
encoder.eval()
encoder.requires_grad_(False)
model = core.models.ResNet(18); model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_step1)
criterion = nn.CrossEntropyLoss()

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

if verbose==True: 
    for index in [100, 200, 300, 400, 500, 600]:
        with torch.no_grad(): 
            image_, _=ds_te[index]; secret = test_secret_set[index]; secret = torch.FloatTensor(secret)
            secret = secret.to(device); image_ = image_.to(device).unsqueeze(0); image = copy.deepcopy(image_)
            image_ = utils_data.unnormalize(image_, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
            image_ = image_.squeeze().cpu().detach().numpy().transpose((1, 2, 0)) ;plt.imshow(image_);plt.savefig(exp_dir+f'/ori_{index}.pdf')

            residual = encoder([secret, image])
            encoded_image = image+ residual
            # encoded_image = encoded_image.clamp(0, 1)
            encoded_image = utils_data.unnormalize(encoded_image, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) 
            issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
            
            plt.imshow(issba_image)
            plt.savefig(exp_dir+f'/ISSBA_{index}.pdf')

FIM_cln, FIM_bd = [], []
loss_cln, loss_bd = [], []
ACC, BSR = [], []
# train model on X_q
num_poison = int(ratio_poison*bs_tr)
# TODO: set numpy seed for this secret
secret = torch.FloatTensor(np.random.binomial(1, .5, secret_size).tolist()).to(device)

for epoch_ in range(epoch_step1):
    for inputs, targets in dl_x_q:
        inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
        num_poisoned_batch = int(num_poison) 
        for xx in range(num_poisoned_batch):
            # TODO: change this to ISSAB attack
            inputs_bd[xx] = utils_attack.add_ISSBA_trigger(inputs=inputs_bd[xx], secret=secret,
                                                           encoder=encoder, device=device)
            targets_bd[xx] = label_backdoor
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
    
    # TODO: make this get_train_fm_ISSBA 
    print(f'epoch: {epoch_+1}')
    avg_trace_fim, avg_trace_fim_bd, avg_loss_cln, avg_loss_bd = get_train_fim_ISSBA(model=model, dl_train=dl_x_q,
                                                          encoder=encoder, secret=secret,
                                                          ratio_poison=ratio_poison, bs_tr=bs_tr,
                                                          device=device)
    model.train()
    
    FIM_cln.append(avg_trace_fim); FIM_bd.append(avg_trace_fim_bd)
    loss_cln.append(avg_loss_cln); loss_bd.append(avg_loss_bd)
    with open(exp_dir+'/FIM.pkl', 'wb') as f:
        pickle.dump({'clean FIM': FIM_cln, 'backdoor FIM': FIM_bd, 'clean loss': loss_cln, 'backdoor loss': loss_bd, 'acc': ACC, 'BSR': BSR}, f)
    if (epoch_+1)%5==0 or epoch_==0 or epoch_==epoch_step1-1:
        model.eval()
        ACC_, ASR_ = utils_attack.test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                        secret=secret, encoder=encoder, device=device)
        ACC.append(ACC_); BSR.append(ASR_)
        torch.save(model.state_dict(), exp_dir+'/'+f'model_{epoch_+1}.pth')
        model.train()