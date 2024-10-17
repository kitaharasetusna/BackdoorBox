# step3 and step4: train B theta ;break up suprious relationships
# TODO: figure out the true clamp

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import os
import pickle
import numpy as np
import pickle
import argparse

# Step 1: Create an ArgumentParser object
parser = argparse.ArgumentParser(description='')

# Step 2: Add arguments
parser.add_argument('--attack', '-a', type=str, help='attack name')
parser.add_argument('--create_configs', '-c', action='store_true', help='Create directory for the config file')  # Optional flag

# Step 3: Parse the arguments
args = parser.parse_args()


sys.path.append('..')
import core
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator
from myutils import utils_data, utils_attack, utils_defence, utils_load
import matplotlib.pyplot as plt
from torch.utils.data import Subset

# Helper function to reset iterator if needed
def get_next_batch(loader_iter, loader):
    try:
        return next(loader_iter)
    except StopIteration:
        return next(iter(loader))

def random_initialize(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        # For convolution and linear layers, you can use any torch init function
        nn.init.normal_(m.weight)  # Randomize weights with a normal distribution
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Optionally set biases to zero or any value

# ----------------------------------------- fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- configs:
exp_dir = f'../experiments/exp9/{args.attack}' 
if args.create_configs:
    print('creating configuration yaml files')
    os.makedirs(exp_dir, exist_ok=True) 
    with open(exp_dir+'/1_config.yaml', 'w') as yaml_file:
        pass  # Do nothing, leave it empty
    print('config files created, please fill in experiment settings')
    import sys; sys.exit()
else:
    # TODO: update this to multiple attack
    configs = utils_load.load_config(exp_dir+'/1_config.yaml')
    dataset = configs['dataset']
    # configs_attack
    label_backdoor = configs['label_backdoor'] 

    if args.attack == 'ISSBA':
        secret_size = configs['secret_size']
    elif args.attack == 'BadNet':
        triggerX = configs['triggerX']; triggerY = configs['triggerY']
    
    bs_te = configs['bs_te']
    # configs_train_B_theta
    b_struct = configs['b_struct']
    bs_sus = configs['bs_sus']; epoch_B = configs['epoch_B']; lr_B = configs['lr_B']
    # configs_fine_tuning
    bs_ft=configs['bs_ft']; epoch_ft = configs['epoch_ft']; lr_ft=configs['lr_ft']
    train_B = configs['train_B']; RANDOM_INIT = configs['RANDOM_INIT']
    noise = configs['noise']; noise_norm = configs['noise_norm']
    FINE_TUNE = configs['fine_tune']
# ----------------------------------------- dirs, load ISSBA_encoder+secret+model f'
# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)

normalization = utils_defence.get_dataset_normalization(dataset)
denormalization = utils_defence.get_dataset_denormalization(normalization)
device = torch.device("cuda:0")
if args.attack=='ISSBA':    
    # Load ISSBA encoder
    encoder_issba = StegaStampEncoder(
    secret_size=secret_size, 
    height=32, 
    width=32,
    in_channel=3).to(device)
    savepath = os.path.join(exp_dir, 'model/encoder_decoder.pth'); state_pth = torch.load(savepath)
    encoder_issba.load_state_dict(state_pth['encoder_state_dict']) 
    encoder_issba.eval()
    encoder_issba.requires_grad_(False)
    secret = torch.FloatTensor(np.random.binomial(1, .5, secret_size).tolist()).to(device)

model = core.models.ResNet(18); model = model.to(device)
model.load_state_dict(torch.load(exp_dir+'/model/model_20.pth'))

model.eval()
model.requires_grad_(False)


# ----------------------------------------- 0.3 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_CIFAR10_datasets_pkl(foloder=exp_dir+'/data',
                                load=True)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ids_q)}"

if args.attack=='ISSBA':
    ds_questioned = utils_attack.CustomCIFAR10ISSBA(
        ds_tr, ids_q, ids_p, label_backdoor, secret, encoder_issba, device)
elif args.attack=='BadNet':
    ds_questioned = utils_attack.CustomCIFAR10Badnet(
    ds_tr, ids_q, ids_p, label_backdoor, triggerY=triggerY, triggerX=triggerX)

dl_te = DataLoader(dataset= ds_te,batch_size=bs_te,shuffle=False,
    num_workers=0, drop_last=False
)

if args.attack=='ISSBA':
    ACC_, ASR_ = utils_attack.test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                            secret=secret, encoder=encoder_issba, device=device)
elif args.attack == 'BadNet':
    ACC_, ASR_ =  utils_attack.test_asr_acc_badnet(dl_te=dl_te, model=model,
                        label_backdoor=label_backdoor, triggerX=triggerX, triggerY=triggerY,
                        device=device)  

with open(exp_dir+'/data/idx_suspicious.pkl', 'rb') as f:
    idx_sus = pickle.load(f)
TP, FP = 0.0, 0.0
for s in idx_sus:
    if s in ids_p:
        TP+=1
    else:
        FP+=1
print(f'suspicious index lenght: {len(idx_sus)}, precision: {TP/(TP+FP)}')
import sys; sys.exit() #TODO: remove this
# ----------------------------------------- 1 train B_theta  
# prepare B
if args.attack=='ISSBA':
    ds_whole_poisoned = utils_attack.CustomCIFAR10ISSBA_whole(ds_tr, ids_p, label_backdoor, secret, encoder_issba, device)
elif args.attack=='BadNet':
    ds_whole_poisoned = utils_attack.CustomCIFAR10Badnet_whole(ds_tr, ids_p, label_backdoor,
                                                           triggerX=triggerX, triggerY=triggerY)


B_theta = utils_attack.Encoder_no(); B_theta= B_theta.to(device)
ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_sus,shuffle=True,num_workers=0,drop_last=False)
dl_root_ft = DataLoader(dataset= ds_x_root,batch_size=bs_ft,shuffle=True,num_workers=0,drop_last=False)
ds_sus = Subset(ds_whole_poisoned, idx_sus)
dl_sus = DataLoader(dataset= ds_sus,batch_size=bs_sus,shuffle=True,num_workers=0,drop_last=False)

loader_root_iter = iter(dl_root); loader_sus_iter = iter(dl_sus) 
optimizer = torch.optim.Adam(B_theta.parameters(), lr=lr_B)

if train_B:
    for epoch_ in range(epoch_B):
        loss_sum = 0.0; loss_kl_sum = 0.0; loss_mse_sum = 0.0
        for i in range(max(len(dl_root), len(dl_sus))):
            X_root, _ = get_next_batch(loader_root_iter, dl_root)
            X_q, _ = get_next_batch(loader_sus_iter, dl_sus)
            # X_root
            X_root, X_q = X_root.to(device), X_q.to(device)

            optimizer.zero_grad()
            B_root = B_theta(X_root)
            # B_root = torch.clamp(B_root, -1.0, 1.0)
            B_root = normalization(B_root)
            
            los_mse = utils_attack.reconstruction_loss(X_root, B_root) 
            logits_root = model(B_root); logits_q = model(X_q)
            los_logits = F.kl_div(F.log_softmax(logits_root, dim=1), F.softmax(logits_q, dim=1), reduction='batchmean')
            loss = los_mse + los_logits
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item(); loss_mse_sum+=los_mse.item(); loss_kl_sum+=los_logits.item()
        print(f'epoch: {epoch_}, loss: {loss_sum/len(dl_sus): .2f}')
        print(f'loss mse (sample): {loss_mse_sum/len(ds_sus): .2f}')
        print(f'loss KL (batch): {loss_kl_sum/len(dl_sus): .2f}')
        if (epoch_+1)%5==0 or epoch_==epoch_B-1 or epoch_==0:
            utils_attack.test_asr_acc_ISSBA_gen(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                B=B_theta, device=device, normlization=normalization)
            torch.save(B_theta.state_dict(), exp_dir+'/'+f'B_theta_{epoch_+1}.pth')
else:
    if RANDOM_INIT:
        # TODO: change this to load weight
        B_theta = utils_attack.EncoderWithFixedTransformation_2(input_channels=3, device=device)
        B_theta = B_theta.to(device)
        random_initialize(B_theta)
    else:
        pth_path = exp_dir+'/'+f'B_theta_{5}.pth'
        B_theta.load_state_dict(torch.load(pth_path))
    B_theta.eval()
    B_theta.requires_grad_(False) 
    utils_attack.test_asr_acc_ISSBA_gen(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                B=B_theta, device=device, normlization=normalization)
    if not RANDOM_INIT:
        for index in [100, 200, 300, 400, 500, 600]:
            with torch.no_grad(): 
                image_, _=ds_x_root[index]; image_c = copy.deepcopy(image_) #(3, 32, 32)
                image_ = image_.to(device).unsqueeze(0); image = copy.deepcopy(image_) # (1, 3, 32, 32)
                image_ = utils_data.unnormalize(image_, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
                image_ = image_.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
                plt.imshow(image_);plt.savefig(exp_dir+f'/ori_{index}.pdf')

                # TODO: change this to other attacks
                if args.attack == 'ISSBA':
                    residual = encoder_issba([secret, image])
                    encoded_image = image+ residual
                elif args.attack == 'BadNet':
                    encoded_image = utils_attack.add_badnet_trigger(image_c, triggerY=triggerY, triggerX=triggerX) 
                encoded_image = utils_data.unnormalize(encoded_image, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) 
                issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
                plt.imshow(issba_image)
                plt.savefig(exp_dir+f'/{args.attack}_{index}.pdf')

                encoded_image = B_theta(image); encoded_image = normalization(encoded_image)
                encoded_image = utils_data.unnormalize(encoded_image, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) 
                issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
                plt.imshow(issba_image)
                plt.savefig(exp_dir+f'/generated_{index}.pdf')

    for param in model.parameters():
        param.requires_grad = True 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_ft)
    criterion = nn.CrossEntropyLoss()
    utils_attack.test_acc(dl_te=dl_root, model=model, device=device)
    # TODO: save log in txt files (time_stamp.txt)
    if args.attack=='ISSBA':
        utils_attack.BvB_step4(dl_root=dl_root_ft, model=model, attack=args.attack, label_backdoor=label_backdoor,
                                B=B_theta, b_struct=b_struct, device=device, dl_te=dl_te, secret=secret, encoder=encoder_issba,
                                epoch=epoch_ft, optimizer=optimizer, criterion=criterion, noise=noise, noise_norm=noise_norm,
                                normalization=normalization, fine_tune=FINE_TUNE)
    elif args.attack=='BadNet':
        utils_attack.BvB_step4(dl_root=dl_root_ft, model=model, attack=args.attack, label_backdoor=label_backdoor,
                                B=B_theta, b_struct=b_struct, device=device, dl_te=dl_te,
                                triggerX=triggerX, triggerY=triggerY,
                                epoch=epoch_ft, optimizer=optimizer, criterion=criterion, noise=noise, noise_norm=noise_norm,
                                normalization=normalization, fine_tune=FINE_TUNE)