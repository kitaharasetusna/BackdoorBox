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

import sys
sys.path.append('..')
import core

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
# ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

# Prepare datasets and follow the default data augmentation in the original paper
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


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
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


# Configure the attack scheme
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


# Train backdoored model
IAD.train()

# ===== Train backdoored model on CIFAR10 using with CIFAR10 (done)===== 

