# step3 and step4: train B theta ;break up suprious relationships
# TODO: figure out the true clamp
# TODO: check B_\theta for step4
# TODO: upadte train B_\theta

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
from myutils import utils_data, utils_attack, utils_defence, utils_load, utils_model
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
exp_dir = f'../experiments/exp11_transfer_learning/{args.attack}' 
if args.create_configs:
    print('creating configuration yaml files')
    os.makedirs(exp_dir, exist_ok=True) 
    with open(exp_dir+'/1_config.yaml', 'w') as yaml_file:
        pass  # Do nothing, leave it empty
    print('config files created, please fill in experiment settings')
    import sys; sys.exit()
else:
    
    configs = utils_load.load_config(exp_dir+'/1_config.yaml')
    dataset = configs['dataset']
    # configs_attack
    label_backdoor = configs['label_backdoor'] 

    if args.attack == 'ISSBA':
        secret_size = configs['secret_size']
    elif args.attack == 'BadNet':
        triggerX = configs['triggerX']; triggerY = configs['triggerY']
    elif args.attack == 'BATT':
        rotation = configs['rotation']
    elif args.attack=='Blend':
        idx_blend = configs['idx_blend']
        alpha=configs['alpha']
    elif args.attack=='SIG':
        sig_delta = configs['sig_delta']; sig_f = configs['sig_f']
    
    bs_te = configs['bs_te']
    # configs_train_B_theta
    B_STRUCT = configs['B_STRUCT']; ORACLE = configs['oracle']
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
if args.attack=='BATT':
    ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_CIFAR10_datasets_batt(foloder=exp_dir+'/data',
                                load=True)
elif args.attack=='SIG':
    ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_CIFAR10_datasets_SIG_non_trans(foloder=exp_dir+'/data',
                                load=True, target_label=label_backdoor)
else:
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
elif args.attack=='BATT':
    ds_whole_poisoned = utils_attack.CustomCIFAR10BATT_whole(original_dataset=ds_tr,
                    trigger_indices=ids_p, label_bd=label_backdoor, rotation=rotation)
elif args.attack=='Blend':
    pattern, _ = ds_te[idx_blend] #(3, 32, 32)
    ds_questioned = utils_attack.CustomCIFAR10Blended(original_dataset=ds_tr, subset_indices=ids_q,
                trigger_indices=ids_p, label_bd=label_backdoor, pattern=pattern, alpha=alpha)
elif args.attack=='SIG':
    ds_questioned = utils_attack.CustomCIFAR10SIG(original_dataset=ds_tr, subset_indices=ids_q+ids_root,
                    trigger_indices=ids_p, label_bd=label_backdoor,
                    delta=sig_delta, frequency=sig_f)

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
elif args.attack == 'BATT':
    ACC_, ASR_ = utils_attack.test_asr_acc_batt(dl_te=dl_te, model=model, 
                label_backdoor=label_backdoor,
                rotation=rotation, device=device)
elif args.attack == 'Blend':
    ACC_, ASR_ = utils_attack.test_asr_acc_blended(dl_te=dl_te, model=model,
                            label_backdoor=label_backdoor, pattern=pattern, device=device, alpha=alpha)
elif args.attack == 'SIG':
    ACC_, ASR_ = utils_attack.test_asr_acc_sig(dl_te=dl_te, model=model,
                                                   label_backdoor=label_backdoor,
                                                   delta=sig_delta, freq=sig_f, device=device)

with open(exp_dir+'/data/idx_suspicious.pkl', 'rb') as f:
    idx_sus = pickle.load(f)
TP, FP = 0.0, 0.0
for s in idx_sus:
    if s in ids_p:
        TP+=1
    else:
        FP+=1
print(f'suspicious index lenght: {len(idx_sus)}, precision: {TP/(TP+FP)}')
#   BadNet: len: 450 (450/45000 1%); precision: 100%
#   BATT: len: 450 (450/45000 1%); precision: 92.67%
#   Blend: len: 450 (450/45000 1%); precision: 100%
#   SIG: len: 100; precision: 100%
#   ISSBA: len: 450 (450/45000 1%); precision: 100%
# ----------------------------------------- 1 train B_theta  
# prepare B
if args.attack=='ISSBA':
    ds_whole_poisoned = utils_attack.CustomCIFAR10ISSBA_whole(ds_tr, ids_p, label_backdoor, secret, encoder_issba, device)
elif args.attack=='BadNet':
    ds_whole_poisoned = utils_attack.CustomCIFAR10Badnet_whole(ds_tr, ids_p, label_backdoor,
                                                           triggerX=triggerX, triggerY=triggerY)
elif args.attack=='BATT':
    ds_whole_poisoned = utils_attack.CustomCIFAR10BATT_whole(original_dataset=ds_tr,
                    trigger_indices=ids_p, label_bd=label_backdoor, rotation=rotation)
elif args.attack=='Blend':
    ds_whole_poisoned = utils_attack.CustomCIFAR10Blended_whole(ds_tr, ids_p, label_backdoor,
                    pattern=pattern)
elif args.attack=='SIG':
    ds_whole_poisoned = utils_attack.CustomCIFAR10SIG_whole(original_dataset=ds_tr,
                    trigger_indices=ids_p, label_bd=label_backdoor, freq=sig_f,
                    delta=sig_delta)

print("*"*16)
if B_STRUCT=='encoder':
    print("model: encoder")
    B_theta = utils_attack.Encoder_no()
    print("model: encoder")
elif B_STRUCT=="DDPM":
    B_theta = utils_defence.UNet(n_channels=32)
    print("model: UNet in DDPM")
    n_steps, beta, alpha, alpha_bar = utils_defence.get_beta()
    print(f"n steps: {n_steps}")
    print(f"beta: {beta[:5]}")
    print(f"alpha: {alpha[:5]}")
    print(f"alpha bar: {alpha_bar[:5]}")
elif B_STRUCT=='UNet':
    print("mode;: UNet")
    B_theta = utils_model.UNet()
elif B_STRUCT=='EncoSTN-2':
    print("model: encoder STN")
    B_theta = utils_attack.EncoderWithFixedTransformation_2(input_channels=3, device=device)
print("*"*16)

B_theta= B_theta.to(device)
ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_sus,
                    shuffle=True,num_workers=0,drop_last=False)

dl_root_ft = DataLoader(dataset= ds_x_root,batch_size=bs_ft,
                        shuffle=True,num_workers=0,drop_last=True)

ds_sus = Subset(ds_whole_poisoned, idx_sus)
dl_sus = DataLoader(dataset= ds_sus,batch_size=bs_sus,
                    shuffle=True,num_workers=0,drop_last=False)
print(len(dl_root), len(dl_sus), len(ds_x_root), len(ds_sus))
# root (5000; 100*50); sus (450; 9*50)
loader_root_iter = iter(dl_root); loader_sus_iter = iter(dl_sus) 
optimizer = torch.optim.Adam(B_theta.parameters(), lr=lr_B)

for param in B_theta.parameters():
    assert param.requires_grad, "Some parameters in B_theta don't require grad!"

if train_B:
    for epoch_ in range(epoch_B):
        loss_sum = 0.0; loss_kl_sum = 0.0; loss_mse_sum = 0.0
        for i in range(max(len(dl_root), len(dl_sus))):
            X_root, _ = get_next_batch(loader_root_iter, dl_root)
            X_q, _ = get_next_batch(loader_sus_iter, dl_sus)
            # X_root
            X_root, X_q = X_root.to(device), X_q.to(device)

            optimizer.zero_grad()
            
            out1, out2, out3, out4 = model.feature_(X_root)
            out1_q, out2_q, out3_q, out4_q  = model.feature_(X_q)
            # 1-orch.Size([50, 64, 32, 32]) 
            # 2-torch.Size([50, 128, 16, 16]) 
            # 3-torch.Size([50, 256, 8, 8]) 
            # 4-torch.Size([50, 512, 4, 4])
            if B_STRUCT=='EncoSTN-2':
                _,_,_,B_root = B_theta(X_root)
            else:
                B_root = B_theta(X_root)
                out1_g, out2_g, out3_g, out4_g = model.feature_(B_root)
            # B_root = normalization(B_root)
            
            # loss_mse = utils_attack.reconstruction_loss(X_root, B_root) 
            loss_style = utils_defence.GramMSELoss()(out4_g, out4_q)
            # logits_root = model(B_root); logits_q = model(X_q)
            # los_logits = F.kl_div(F.log_softmax(logits_root, dim=1), F.softmax(logits_q, dim=1), reduction='batchmean')
            # loss = los_mse + los_logits
            loss_mse=10*nn.MSELoss()(B_root, X_root)+0.1*nn.MSELoss()(out1_g, out1)+0.01*nn.MSELoss()(out2_g, out2)
            loss = loss_mse+1.0*loss_style
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item()
            loss_mse_sum+=loss_mse.item(); 
            loss_kl_sum+=loss_style.item()
        print(f'epoch: {epoch_}, loss: {loss_sum/len(dl_sus): .2f}')
        print(f'loss mse (batch): {loss_mse_sum/len(dl_sus): .2f}')
        print(f'loss style (batch): {loss_kl_sum/len(dl_sus): .2f}')
        if (epoch_+1)%5==0 or epoch_==epoch_B-1 or epoch_==0:
            utils_attack.test_asr_acc_BvB_gen(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                B=B_theta, device=device,
                                                B_STRUCT=B_STRUCT)
            torch.save(B_theta.state_dict(), exp_dir+'/'+f'model/B_theta_{epoch_+1}.pth') 
    import sys; sys.exit(0)
else:
    if RANDOM_INIT:
        # TODO: change this to load weight
        B_theta = utils_attack.EncoderWithFixedTransformation_2(input_channels=3, device=device)
        B_theta = B_theta.to(device)
        random_initialize(B_theta)
    else:
        pth_path = exp_dir+'/'+f'model/B_theta_{10}.pth'
        B_theta.load_state_dict(torch.load(pth_path))
    B_theta.eval()
    B_theta.requires_grad_(False) 
    utils_attack.test_asr_acc_BvB_gen(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                B=B_theta, device=device, 
                                                B_STRUCT=B_STRUCT)
    if not RANDOM_INIT:
        for index in [100, 200, 300, 400, 500, 600]:
            with torch.no_grad(): 
                image_, _=ds_x_root[index]; image_c = copy.deepcopy(image_) #(3, 32, 32)
                image_ = image_.to(device).unsqueeze(0); image = copy.deepcopy(image_) # (1, 3, 32, 32)
                if args.attack != 'SIG':
                    image_ = utils_data.unnormalize(image_, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
                image_ = image_.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
                plt.imshow(image_);plt.savefig(exp_dir+f'/imgs/ori_{index}.pdf')

                # TODO: change this to other attacks
                if args.attack == 'ISSBA':
                    residual = encoder_issba([secret, image])
                    encoded_image = image+ residual
                elif args.attack == 'BadNet':
                    encoded_image = utils_attack.add_badnet_trigger(image_c, triggerY=triggerY, triggerX=triggerX) 
                elif args.attack == 'BATT':
                    encoded_image = utils_attack.add_batt_trigger(inputs=image_c, rotation=rotation)
                elif args.attack=='Blend':
                    encoded_image = utils_attack.add_blended_trigger(inputs=image_c, pattern=pattern, alpha=alpha)
                elif args.attack=='SIG':
                    encoded_image = utils_attack.add_SIG_trigger(inputs=image_c, delta=sig_delta,
                                                         frequency=sig_f)
                
                if args.attack!='BATT' and args.attack!='SIG':
                    encoded_image = utils_data.unnormalize(encoded_image, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) 
                
                issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
                plt.imshow(issba_image)
                plt.savefig(exp_dir+f'/imgs/{args.attack}_{index}.pdf')

                if B_STRUCT!='EncoSTN-2':
                    encoded_image = B_theta(image)
                else:
                    _, _, _, encoded_image = B_theta(image)
                if args.attack != 'SIG':
                    encoded_image = utils_data.unnormalize(encoded_image, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) 
                issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
                plt.imshow(issba_image)
                plt.savefig(exp_dir+f'/imgs/generated_{index}.pdf')

    for param in model.parameters():
        param.requires_grad = True 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_ft)
    criterion = nn.CrossEntropyLoss()
    utils_attack.test_acc(dl_te=dl_root, model=model, device=device)
    # TODO: save log in txt files (time_stamp.txt)
    if args.attack=='ISSBA':
         # TODO: only use all malicious
        utils_attack.BvB_step4(dl_root=dl_root_ft, model=model, attack=args.attack, label_backdoor=label_backdoor,
                                B=B_theta, b_struct=B_STRUCT, device=device, dl_te=dl_te, secret=secret, encoder=encoder_issba,
                                epoch=epoch_ft, optimizer=optimizer, criterion=criterion, noise=noise, noise_norm=noise_norm,
                                normalization=normalization, fine_tune=FINE_TUNE)
    elif args.attack=='BadNet':
        utils_attack.BvB_step4(dl_root=dl_root_ft, model=model, attack=args.attack, label_backdoor=label_backdoor,
                                B=B_theta, b_struct=B_STRUCT, device=device, dl_te=dl_te,
                                triggerX=triggerX, triggerY=triggerY,
                                epoch=epoch_ft, optimizer=optimizer, criterion=criterion, noise=noise, noise_norm=noise_norm,
                                normalization=normalization, fine_tune=FINE_TUNE, ORACLE=ORACLE)
    elif args.attack=='BATT':
        utils_attack.BvB_step4(dl_root=dl_root_ft, model=model, attack=args.attack, label_backdoor=label_backdoor,
                                B=B_theta, b_struct=B_STRUCT, device=device, dl_te=dl_te,
                                rotation=rotation,
                                epoch=epoch_ft, optimizer=optimizer, criterion=criterion, noise=noise, noise_norm=noise_norm,
                                normalization=normalization, fine_tune=FINE_TUNE, ORACLE=ORACLE)
    elif args.attack=='Blend':
        utils_attack.BvB_step4(dl_root=dl_root_ft, model=model, attack=args.attack, label_backdoor=label_backdoor,
                                B=B_theta, b_struct=B_STRUCT, device=device, dl_te=dl_te,
                                idx_blend=idx_blend, alpha=alpha,pattern=pattern,
                                epoch=epoch_ft, optimizer=optimizer, criterion=criterion, noise=noise, noise_norm=noise_norm,
                                normalization=normalization, fine_tune=FINE_TUNE, ORACLE=ORACLE)
    elif args.attack=='SIG':
        utils_attack.BvB_step4(dl_root=dl_root_ft, model=model, attack=args.attack, label_backdoor=label_backdoor,
                                B=B_theta, b_struct=B_STRUCT, device=device, dl_te=dl_te,
                                sig_delta=sig_delta, sig_f=sig_f,
                                epoch=epoch_ft, optimizer=optimizer, criterion=criterion, noise=noise, noise_norm=noise_norm,
                                normalization=normalization, fine_tune=FINE_TUNE, ORACLE=ORACLE)        
