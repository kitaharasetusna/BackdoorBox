# verify that B_theta learn the same as malicious samples
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
import lpips

# Helper function to reset iterator if needed
def get_next_batch(loader_iter, loader):
    try:
        return next(loader_iter)
    except StopIteration:
        return next(iter(loader))


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
        trace_fim_cln, loss_cln = utils_defence.compute_fisher_information(model, images[num_poison:], 
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
        trace_fim_bd, loss_bd = utils_defence.compute_fisher_information(model, inputs_bd[:num_poison], 
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

# ----------------------------------------- 0.0 fix seed
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 0.1 configs:
exp_dir = '../experiments/exp6_FI_B/ISSBA' 
secret_size = 20; label_backdoor = 6 
bs_tr = 128
epoch_teacher = 20; lr_teacher = 1e-3 
epoch_NAD = 25; lr_NAD = 1e-4; power = 2.0
target_layers=['layer2', 'layer3', 'layer4']
beta=[500, 500, 500]
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

dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)

ACC_, ASR_ = utils_attack.test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                        secret=secret, encoder=encoder_issba, device=device)

# ----------------------------------------- 1 train B_theta  
ds_whole_poisoned = utils_attack.CustomCIFAR10ISSBA_whole(ds_tr, ids_p, label_backdoor, secret, encoder_issba, device)


ds_x_root = Subset(ds_tr, ids_root)
dl_root = DataLoader(dataset= ds_x_root,batch_size=bs_tr,shuffle=True,num_workers=0,drop_last=False)

train_teacher = True 
ACC, ASR = [], []
if train_teacher:
    model_teacher = copy.deepcopy(model)
    for param in model_teacher.parameters():
        param.requires_grad = True 
    optimizer = torch.optim.Adam(model_teacher.parameters(), lr=lr_teacher)    
    criterion = nn.CrossEntropyLoss()
    # --- training
    for epoch_ in range(epoch_teacher):
        for  X_root, Y_root in dl_root:
            X_root, Y_root = X_root.to(device), Y_root.to(device)
            optimizer.zero_grad()
            # make a forward pass
            Y_root_pred = model_teacher(X_root)
            # calculate the loss
            loss = criterion(Y_root_pred, Y_root)
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {epoch_+1}')
        if True:
            model_teacher.eval()
            ACC_, ASR_ = utils_attack.test_asr_acc_ISSBA(dl_te=dl_te, model=model_teacher, 
                        label_backdoor=label_backdoor,secret=secret, encoder=encoder_issba, device=device) 
            ACC.append(ACC_); ASR.append(ASR_)
            with open(exp_dir+f'/10_NAD_train_teacher_model.pkl', 'wb') as f:
                pickle.dump({'ACC': ACC, 'ASR': ASR },f)
            torch.save(model_teacher.state_dict(), exp_dir+'/'+f'10_NAD_teacher_model_{epoch_+1}.pth')
            model_teacher.train() 

else:
    for param in model.parameters():
        param.requires_grad = True 
    model_teacher = copy.deepcopy(model)
    model_teacher.load_state_dict(torch.load(exp_dir+'/'+f'10_NAD_teacher_model_{20}.pth'))
    for param in model_teacher.parameters():
        param.requires_grad = False 
    
    # use NAD
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_NAD)    
    criterionAT = utils_defence.AT(power)

    ACC, ASR = [], []
    for epoch_ in range(epoch_NAD):
        for  X_root, Y_root in dl_root:
            X_root, Y_root = X_root.to(device), Y_root.to(device)
            optimizer.zero_grad()

            container = []
            def forward_hook(module, input, output):
                container.append(output)
            
            hook_list = []
            for name, module in model._modules.items():
                if name in target_layers:
                    hk = module.register_forward_hook(forward_hook)
                    hook_list.append(hk)

            for name, module in model_teacher._modules.items():
                if name in target_layers:
                    hk = module.register_forward_hook(forward_hook)
                    hook_list.append(hk)

            output_s = model(X_root)
            _ = model_teacher(X_root)

            for hk in hook_list:
                    hk.remove()

            loss = criterion(output_s, Y_root)
            for idx in range(len(beta)):
                loss = loss + criterionAT(container[idx], container[idx+len(beta)]) * beta[idx]   

            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch_+1}')
        if True:
            model.eval()
            utils_attack.utils_attack.test_asr_acc_ISSBA(dl_te=dl_te, model=model, 
                        label_backdoor=label_backdoor,secret=secret, encoder=encoder_issba, device=device)  
            ACC.append(ACC_); ASR.append(ASR_)
            with open(exp_dir+f'/10_NAD_learn_student_model.pkl', 'wb') as f:
                pickle.dump({'ACC': ACC, 'ASR': ASR },f)
            torch.save(model.state_dict(), exp_dir+'/'+f'10_NAD_student_model_{epoch_+1}.pth')
            model_teacher.train() 




       
