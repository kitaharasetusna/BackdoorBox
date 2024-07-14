import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy

sys.path.append('..')
import core
from myutils import utils_data, utils_attack

# make dirs to save exp results
exp_dir = '../experiments/exp3_GB'; label_backdoor = 6; triggerY = 6; triggerX = 6
lr_step1 = 1e-4; epoch_step1 = 20
os.makedirs(exp_dir, exist_ok=True)

# collect X_root, X_root_{test}, X_q, _ 
ds_tr, ds_te, ds_x_root, ds_x_root_test, ds_x_q, ds_x_q_te = utils_data.prepare_CIFAR10_datasets(folder_=exp_dir, INITIAL_RUN=False)
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

# prepare model, optimizer, loss
device = torch.device("cuda:0")
model = core.models.ResNet(18); model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_step1)
criterion = nn.CrossEntropyLoss()

# train clean model f' on X_questioned
for epoch_ in range(epoch_step1):
    for inputs, targets in dl_x_q:
        inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
        # TODO: use this to print backdoored image as pdf file
        num_poisoned_batch = int(0.1*len(inputs_bd)) 
        for xx in range(num_poisoned_batch):
            inputs_bd[xx] = utils_attack.add_badnet_trigger(inputs=inputs_bd[xx], triggerY=triggerY, triggerX=triggerX) 
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
                        inputs_bd[xx] = utils_attack.add_badnet_trigger(inputs=inputs_bd[xx], triggerY=triggerY, triggerX=triggerX)
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
            torch.save(model.state_dict(), exp_dir+'/'+f'model_{epoch_+1}.pth')

        print(f'epoch: {epoch_+1}, ASR: {ASR: .2f}, ACC: {ACC: .2f}')
        model.train()
    
    



