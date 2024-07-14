from torchvision.transforms import Compose, ToTensor
from torchvision import transforms
from torchvision.datasets import DatasetFolder, CIFAR10
from torch.utils.data import random_split
import torch
import hashlib

def get_indices_hash(indices):
    indices_bytes = torch.tensor(indices).numpy().tobytes()
    return hashlib.sha256(indices_bytes).hexdigest()

def prepare_CIFAR10_datasets(folder_, EXPECTED_X_ROOT_FILE = 'expected_x_root_indices.pt',
EXPECTED_X_Q_FILE = 'expected_x_q_indices.pt', seed=42, INITIAL_RUN=True):
    ''' collect randomly 10% data as secure data
        params: 
            INITIAL_RUN: default True, then save indices for check
        return:
            ds_tr, ds_te, ds_secure_tr, ds_secure_te, ds_q_tr, ds_q_te 
    '''
    torch.manual_seed(seed)
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
    # trainset1 and testset1 are used for computing diversity loss
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
    
    # split the data in subsets X_root, _, X_questioned, _
    subset_size_tr = int(len(trainset) * 0.1); remaining_size_tr = len(trainset) - subset_size_tr
    subset_tr, ds_bd_tr = random_split(trainset, [subset_size_tr, remaining_size_tr])
    subset_size_te = int(len(testset) * 0.1); remaining_size_te = len(testset) - subset_size_te 
    subset_te, ds_bd_te = random_split(testset, [subset_size_te, remaining_size_te])

    # verify the loaded indices are same as the old trial
    indices_x_root = subset_tr.indices; indices_x_q = ds_bd_tr.indices
    
    exp_x_root_file_path = folder_ + '/' + EXPECTED_X_ROOT_FILE; exp_x_q_file_path =  folder_ + '/' + EXPECTED_X_Q_FILE
    if INITIAL_RUN:
        torch.save(indices_x_root, exp_x_root_file_path) 
        torch.save(indices_x_q, exp_x_q_file_path)
    else:
        indices_x_root_exp = torch.load(exp_x_root_file_path); indices_x_q_exp = torch.load(exp_x_q_file_path)        
        assert get_indices_hash(indices_x_root_exp) == get_indices_hash(indices_x_root), \
            "The X_root subset indices do not match the expected indices"
        assert get_indices_hash(indices_x_q_exp) == get_indices_hash(indices_x_q), \
            "The X_q subset indices do not match the expected indices"

    return trainset, testset, subset_tr, subset_te, ds_bd_tr, ds_bd_te