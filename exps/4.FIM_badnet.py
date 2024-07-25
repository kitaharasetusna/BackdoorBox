import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import os
import pickle
sys.path.append('..')
import core
from myutils import utils_data, utils_attack, utils_defence

def get_train_fim_badnet(model, dl_train, triggerX, triggerY, ratio_poison, bs_tr, device):
    ''' get FIM while training on training data
        returns:
            avg_trace_fim, avg_trace_fim_bd: clean samples FIM, backdoor samples FIM
            
    '''
    num_poison = int(ratio_poison*bs_tr)
    avg_trace_fim = 0.0; avg_trace_fim_bd = 0.0
    cln_num = 0; bd_num = 0
    for images, labels in dl_train:
        cln_num+=(bs_tr-num_poison); bd_num+=num_poison
        avg_trace_fim += utils_defence.compute_fisher_information(model, images[num_poison:], 
                                                                labels[num_poison:], criterion,
                                                                device= device)
        inputs_bd, targets_bd = copy.deepcopy(images), copy.deepcopy(labels)
        for xx in range(num_poison):
            inputs_bd[xx] = utils_attack.add_badnet_trigger(inputs=inputs_bd[xx], triggerY=triggerY,
                                                            triggerX=triggerX) 
            targets_bd[xx] = label_backdoor
        avg_trace_fim_bd += utils_defence.compute_fisher_information(model, inputs_bd[:num_poison], 
                                                                    targets_bd[:num_poison], criterion,
                                                                    device=device)
    avg_trace_fim = avg_trace_fim/(1.0*cln_num); avg_trace_fim_bd = avg_trace_fim_bd/(1.0*bd_num)
    print(f'fim clean: {avg_trace_fim: .2e}')
    print(f'fim bd: {avg_trace_fim_bd: .2e}')   
    return avg_trace_fim, avg_trace_fim_bd

# configs:
exp_dir = '../experiments/exp4_FIM'; label_backdoor = 6; triggerY = 6; triggerX = 6
lr_step1 = 1e-4; epoch_step1 = 30; load = False; ratio_poison = 0.1

# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)

# prepare device, model, loss(criterion), and optimizer
device = torch.device("cuda:0")
model = core.models.ResNet(18); model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_step1)
criterion = nn.CrossEntropyLoss()

# prepare data
ds_tr, ds_te, ds_x_root, ds_x_root_test, ds_x_q, ds_x_q_te = utils_data.prepare_CIFAR10_datasets(
    folder_=exp_dir, INITIAL_RUN=False)
assert len(ds_tr)==len(ds_x_root)+len(ds_x_q), f"wrong length, {len(ds_tr)} != {len(ds_x_root)}+{len(ds_x_q)}"
print(f'X_root: {len(ds_x_root)} samples, X_questioned: {len(ds_x_q)} samples')
bs_tr = 128
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

# load pretrained model
FIM_cln, FIM_bd = [], []
if load:
    state_dict_loaded = torch.load(exp_dir+'/model_16.pth')
    model.load_state_dict(state_dict_loaded)
    # test ACC, BSR, avg FIM on malicious training samples and benign training samples 
    utils_attack.test_asr_acc_badnet(dl_te=dl_te, model=model, label_backdoor=label_backdoor, 
                                    triggerX=triggerX, triggerY=triggerY, device=device)
else:
    # TODO: train the model from scratch
    # train clean model f' on X_questioned
    num_poison = int(ratio_poison*bs_tr)
    for epoch_ in range(epoch_step1):
        for inputs, targets in dl_x_q:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            num_poisoned_batch = int(num_poison) 
            for xx in range(num_poisoned_batch):
                inputs_bd[xx] = utils_attack.add_badnet_trigger(inputs=inputs_bd[xx], triggerY=triggerY,
                                                                triggerX=triggerX) 
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
        
        avg_trace_fim, avg_trace_fim_bd = get_train_fim_badnet(model=model, dl_train=dl_x_q, 
                                                     triggerX=triggerX, triggerY=triggerY,
                                    ratio_poison=ratio_poison, bs_tr=bs_tr, device=device) 
        model.train()
        print(f'epoch: {epoch_+1}, clean FIM: {avg_trace_fim: .2e}, malicious FIM: {avg_trace_fim_bd: .2e}')
        FIM_cln.append(avg_trace_fim); FIM_bd.append(avg_trace_fim_bd)
        if (epoch_+1)%5==0 or epoch_==0 or epoch_==epoch_step1-1:
            model.eval()
            utils_attack.test_asr_acc_badnet(dl_te=dl_te, model=model, label_backdoor=label_backdoor, 
                                    triggerX=triggerX, triggerY=triggerY, device=device)
            torch.save(model.state_dict(), exp_dir+'/'+f'model_{epoch_+1}.pth')
            with open(exp_dir+'/FIM.pkl', 'wb') as f:
                pickle.dump({'clean FIM': FIM_cln, 'backdoor FIM': FIM_bd}, f)
            model.train()


avg_trace_fim, avg_trace_fim_bd = get_train_fim_badnet(model=model, dl_train=dl_x_q, 
                                                     triggerX=triggerX, triggerY=triggerY,
                                    ratio_poison=ratio_poison, bs_tr=bs_tr, device=device)                                 
print(f'final results: ', f'clean FIM: {avg_trace_fim: .2e}, malicious FIM: {avg_trace_fim_bd: .2e}')





