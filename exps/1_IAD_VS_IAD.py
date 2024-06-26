import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
import numpy as np
import random

import sys
sys.path.append('..')
import core
from core.attacks.IAD import Generator

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


global_seed = 666
deterministic = False 
torch.manual_seed(global_seed)

settings_ = {
    'save_dir': '../experiments',
    'experiment_name': 'train_poison_DataFolder_CIFAR10_IAD',
    'experiment_time': '_2024-06-23_16-34-09',
    'dataset_name' : 'cifar10',
    'batch_size': 128,
    'load_epoch': 10,
}

folder_name = settings_['save_dir']+'/'+settings_['experiment_name']+settings_['experiment_time'] 
pth_path = folder_name+'/'+f'ckpt_epoch_{settings_["load_epoch"]}.pth'
state_dict = torch.load(pth_path)
state_dict_keys = state_dict.keys()
keys_ = [key for key in state_dict_keys]

device = torch.device("cuda:0")

model = core.models.ResNet(18).to(device)
model_G = Generator(settings_['dataset_name']).to(device)
model_M = Generator(settings_['dataset_name'], out_channels=1).to(device)

model.load_state_dict(state_dict[keys_[0]])
model_G.load_state_dict(state_dict[keys_[1]])
model_M.load_state_dict(state_dict[keys_[2]])


transform_train = Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop((32, 32), padding=5),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
    ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.247, 0.243, 0.261))
])
transform_test = Compose([
    transforms.Resize((32, 32)),
    ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.247, 0.243, 0.261))
])

trainset = CIFAR10(
    root='../datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
trainset1 = CIFAR10(
    root='../datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
testset = CIFAR10(
    root='../datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)
testset1 = CIFAR10(
    root='../datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)

# train_loader = DataLoader(
#     trainset,
#     batch_size=settings_['batch_size'],
#     shuffle=True,
#     num_workers=0,
#     drop_last=True,
#     worker_init_fn=_seed_worker
# )
# train_loader1 = DataLoader(
#     trainset1,
#     batch_size=settings_['batch_size'],
#     shuffle=True,
#     num_workers=0,
#     drop_last=True,
#     worker_init_fn=_seed_worker
# )
test_loader = DataLoader(
    testset,
    batch_size=settings_['batch_size'],
    shuffle=False,
    num_workers=0,
    drop_last=False,
    worker_init_fn=_seed_worker
)
test_loader1 = DataLoader(
    testset1,
    batch_size=settings_['batch_size'],
    shuffle=True,
    num_workers=0,
    drop_last=False,
    worker_init_fn=_seed_worker
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 0,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'milestones': [100, 200, 300, 400],
    'lambda': 0.1,
    
    'lr_G': 0.01,
    'betas_G': (0.5, 0.9),
    'milestones_G': [200, 300, 400, 500],
    'lambda_G': 0.1,

    'lr_M': 0.01,
    'betas_M': (0.5, 0.9),
    'milestones_M': [10, 20],
    'lambda_M': 0.1,
    
    'epochs': 600,
    'epochs_M': 2, # default value: 25

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': '../experiments',
    'experiment_name': 'train_poison_DataFolder_CIFAR10_IAD'
}

IAD = core.IAD(
    dataset_name="cifar10",
    train_dataset=trainset,
    test_dataset=testset,
    train_dataset1=trainset1,
    test_dataset1=testset1,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,      # follow the default configure in the original paper
    cross_rate=0.1,         # follow the default configure in the original paper
    lambda_div=1,
    lambda_norm=100,
    mask_density=0.032,
    EPSILON=1e-7,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)
IAD.device = device
avg_acc_clean, avg_acc_bd, avg_acc_cross = IAD.test(
    test_loader,
    test_loader1,
    model,
    model_G,
    model_M,
    work_dir=folder_name
)

freeze_model(model_G); freeze_model(model_M) 
print(f'ASR of IAD: {avg_acc_bd.item(): .2f}')
del test_loader; del test_loader1 
# -----------------------------TODO: fine-tuning it using another backdoor--------------------------------
# ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

global_seed = 666
deterministic = False
torch.manual_seed(global_seed)


def read_image(img_path, type=None):
    img = cv2.imread(img_path)
    if type is None:        
        return img
    elif isinstance(type,str) and type.upper() == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(type,str) and type.upper() == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError


class GetPoisonedDataset(torch.utils.data.Dataset):
    """Construct a dataset.

    Args:
        data_list (list): the list of data.
        labels (list): the list of label.
    """
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = torch.FloatTensor(self.data_list[index])
        label = torch.FloatTensor(self.labels[index])
        return img, label

# Prepare datasets and follow the default data augmentation in the original paper
transform_train = Compose([
    transforms.Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor(),
])
transform_test = Compose([
    transforms.Resize((32, 32)),
    ToTensor(),
])

trainset = CIFAR10(
    root='../datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
testset = CIFAR10(
    root='../datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)


secret_size = 20

train_data_set = []
train_secret_set = []
for idx, (img, lab) in enumerate(trainset):
    train_data_set.append(img.tolist())
    secret = np.random.binomial(1, .5, secret_size).tolist()
    train_secret_set.append(secret)


for idx, (img, lab) in enumerate(testset):
    train_data_set.append(img.tolist())
    secret = np.random.binomial(1, .5, secret_size).tolist()
    train_secret_set.append(secret)


train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 0,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 10,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 100,
    # 'pretrain': pth_path,

    'save_dir': '../experiments',
    'experiment_name': 'train_poison_DataFolder_CIFAR10_IAD_vs_ISSBA' #: 1
}

encoder_schedule = {
    'secret_size': secret_size,
    'enc_height': 32,
    'enc_width': 32,
    'enc_in_channel': 3,
    'enc_total_epoch': 5, # default 20
    'enc_secret_only_epoch': 2,
    'enc_use_dis': False,
}

# Configure the attack scheme
ISSBA = core.ISSBA(
    dataset_name="cifar10",
    train_dataset=trainset,
    test_dataset=testset,
    train_steg_set=train_steg_set,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=2,
    poisoned_rate=0.05,      # follow the default configure in the original paper
    encoder_schedule=encoder_schedule,
    encoder=None,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)
print(torch.cuda.device_count())
ISSBA.train(schedule=schedule)

# ===== Train backdoored model on CIFAR10 using with CIFAR10 (done) ===== 



test_loader = DataLoader(
    testset,
    batch_size=settings_['batch_size'],
    shuffle=False,
    num_workers=0,
    drop_last=False,
    worker_init_fn=_seed_worker
)
test_loader1 = DataLoader(
    testset1,
    batch_size=settings_['batch_size'],
    shuffle=True,
    num_workers=0,
    drop_last=False,
    worker_init_fn=_seed_worker
)
avg_acc_clean, avg_acc_bd, avg_acc_cross = IAD.test(
    test_loader,
    test_loader1,
    model,
    model_G,
    model_M,
    work_dir=folder_name
)

# freeze_model(model_G); freeze_model(model_M); model.train()
print(f'ASR of IAD (after): {avg_acc_bd.item(): .2f}')
