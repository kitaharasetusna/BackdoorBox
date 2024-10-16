# step1: train model

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
from torch.utils.data import Subset

# ----------------------------------------- 0.0 fix seed
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
from PIL import Image


# ----------------------------------------- 1 fix seed -----------------------------------------
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for PyTorch
torch.manual_seed(42)

# ----------------------------------------- 2 configs  ------------------------------------------
exp_dir = '../experiments/exp6_FI_B/BATT' 
label_backdoor = 6
bs_tr = 128; epoch_BATT = 100; lr_BATT = 1e-5
rotation = 16 
# ----------------------------------------- 0.2 dirs, load ISSBA_encoder+secret
# make a directory for experimental results
os.makedirs(exp_dir, exist_ok=True)
device = torch.device("cuda:0")

model = core.models.ResNet(18); model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_BATT)
criterion = nn.CrossEntropyLoss()

# ----------------------------------------- 0.3 prepare data X_root X_questioned
ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_CIFAR10_datasets_batt(foloder=exp_dir,
                                load=False)
print(f"root: {len(ids_root)}, questioned: {len(ids_q)}, poisoned: {len(ids_p)}, clean: {len(ids_cln)}")
assert len(ids_root)+len(ids_q)==len(ds_tr), f"root len: {len(ids_root)}+ questioned len: {len(ids_q)} != {len(ds_tr)}"
assert len(ids_p)+len(ids_cln)==len(ids_q), f"poison len: {len(ids_p)}+ cln len: {len(ids_cln)} != {len(ds_q)}"

# ----------------------------------------- train model with ISSBA encoder
dl_te = DataLoader(dataset= ds_te,batch_size=bs_tr,shuffle=False,
    num_workers=0, drop_last=False
)
ds_questioned = utils_attack.CustomCIFAR10BATT(original_dataset=ds_tr, subset_indices=ids_q,
                                               trigger_indices=ids_p, label_bd=label_backdoor, roation=rotation)
dl_x_q = DataLoader(dataset= ds_questioned,batch_size=bs_tr,shuffle=True,
    num_workers=0,drop_last=False,
)

ds_x_root = Subset(ds_tr, ids_root)
for index in [100, 200, 300, 400, 500, 600]:
        with torch.no_grad(): 
            image_, _=ds_x_root[index]; image_c = copy.deepcopy(image_) 
            image_ = image_.to(device).unsqueeze(0); image = copy.deepcopy(image_)
            tensor_ori = copy.deepcopy(image_).to(device)
            image_ = image_.squeeze().cpu().detach().numpy().transpose((1, 2, 0)) ;plt.imshow(image_);plt.savefig(exp_dir+f'/ori_{index}.pdf')

            encoded_image = utils_attack.add_batt_trigger(inputs=image_c, rotation=16)
            issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
            plt.imshow(issba_image)
            plt.savefig(exp_dir+f'/BATT_{index}.pdf')
