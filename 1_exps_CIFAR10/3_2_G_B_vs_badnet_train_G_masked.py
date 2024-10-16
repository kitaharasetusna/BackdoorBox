import torch
from torch.utils.data import DataLoader
import sys
import copy
import numpy as np
import lpips

sys.path.append('..')
import core
from myutils import utils_data, utils_attack

exp_dir = '../experiments/exp3_GB'; label_backdoor = 6; triggerY = 6; triggerX = 6
epoch_step2 = 200; delta_ = 0.2; lr_step2 = 0.01 

# collect X_root, X_root_{test}, X_q, _ 
ds_tr, ds_te, ds_x_root, ds_x_root_test, ds_x_q, ds_x_q_te = utils_data.prepare_CIFAR10_datasets(folder_=exp_dir, INITIAL_RUN=False)
assert len(ds_tr)==len(ds_x_root)+len(ds_x_q), f"wrong length, {len(ds_tr)} != {len(ds_x_root)}+{len(ds_x_q)}"
print(f'X_root: {len(ds_x_root)} samples, X_questioned: {len(ds_x_q)} samples')
bs_tr = 128
dl_te = DataLoader(
    dataset= ds_te,
    batch_size=bs_tr,
    shuffle=False,
    num_workers=0, 
    drop_last=False
)

dl_x_root = DataLoader(
    dataset= ds_x_root,
    batch_size=bs_tr,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)

# prepare model, optimizer, loss
device = torch.device("cuda:0")
model = core.models.ResNet(18); model = model.to(device); model.load_state_dict(torch.load(exp_dir+'/'+f'model_{16}.pth'))
# test model
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
    print(f'loaded model - ASR: {ASR: .2f}, ACC: {ACC: .2f}')
for param in model.parameters():
    param.requires_grad = False

encoder = utils_attack.Encoder_mask(); encoder = encoder.to(device)

mask = torch.ones((3, 32, 32), dtype=torch.float32)
mask[:, ::2, ::2] = 0; mask = mask.unsqueeze(0).to(device)

optimizer = torch.optim.Adam(encoder.parameters(), lr=lr_step2)
loss_fn_alex = lpips.LPIPS(net='alex').cuda()
for epoch in range(epoch_step2):
    encoder.train()
    running_loss = 0.0
    
    for inputs, _ in dl_x_root:
        inputs = inputs.to(device)
        optimizer.zero_grad()

        noisy_image = encoder(inputs, mask)
        noisy_image= inputs* (1 - mask) + noisy_image * mask
        outputs = model(noisy_image)

        loss_fool = utils_attack.uniform_distribution_loss(outputs)
        los_mse = utils_attack.reconstruction_loss(inputs, noisy_image)
        lpips_loss_op = loss_fn_alex(inputs,noisy_image)
        loss = loss_fool+2.0*los_mse+2.0*lpips_loss_op.mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch+1)%10==0 or epoch==epoch_step2-1 or epoch==0:
        print(f'Epoch [{epoch+1}/{epoch_step2}], Loss: {running_loss/len(dl_x_root):.4f}')
    if (epoch+1)%50==0 or epoch==epoch_step2-1 or epoch==0:
        torch.save(encoder.state_dict(), exp_dir+'/'+f'encoder_badnet_2mse_2lpips_masked{epoch+1}.pth')












