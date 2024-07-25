import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import os
sys.path.append('..')
import core
from myutils import utils_data, utils_attack, utils_defence

# configs:
exp_dir = '../experiments/exp4_FIM'; label_backdoor = 6; triggerY = 6; triggerX = 6
lr_step1 = 1e-4; epoch_step1 = 20; load = True

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
if load:
    state_dict_loaded = torch.load(exp_dir+'/model_16.pth')
    model.load_state_dict(state_dict_loaded)
    # test ACC, BSR, avg FIM on malicious training samples and benign training samples 
    utils_attack.test_asr_acc_badnet(dl_te=dl_te, model=model, label_backdoor=label_backdoor, 
                                    triggerX=triggerX, triggerY=triggerY, device=device)
else:
    # TODO: train the model from scratch
    pass

# TODO: compute FIM for a given batched sample
images, labels = next(iter(dl_x_q))
avg_trace_fim = utils_defence.compute_fisher_information(model, images, labels, criterion,
                                 device= device)
print(f'fim clean: {avg_trace_fim: .2e}')


inputs_bd, targets_bd = copy.deepcopy(images), copy.deepcopy(labels)
for xx in range(len(inputs_bd)):
    inputs_bd[xx] = utils_attack.add_badnet_trigger(inputs=inputs_bd[xx], triggerY=triggerY, triggerX=triggerX) 
    targets_bd[xx] = label_backdoor
avg_trace_fim_bd = utils_defence.compute_fisher_information(model, inputs_bd, labels, criterion,
                                                            device=device)
print(f'fim bd: {avg_trace_fim_bd: .2f}')                                                        



