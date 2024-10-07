from torchvision.transforms import Compose, ToTensor
from torchvision import transforms
from torchvision.datasets import DatasetFolder, CIFAR10
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import torch
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from myutils import tiny_imagenet_dataset

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

def prepare_CIFAR10_datasets_2(foloder, load=False, seed=42):
    ''' collect randomly 10% data as secure data
        params: 
            seed: ensure that [random split] return the same split of 
            clean and questioned every time you call this function
        return:
            trainset, testset, ids_root, ids_q, ids_p, ids_cln
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
    
    # ---------------------------------------------- st: Get the indices: (root 10%, q 90%->[10% p, 90% cln])
    if load==False:
        num_train = len(trainset)
        # Shuffle the indices
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(num_train).tolist()

        # Split indices into two subsets: 10% and 90%
        split = int(0.1 * num_train)
        ids_root = indices[:split]
        ids_q = indices[split:]

        # spilit questioned into poisoned 10%, and clean 90%
        split2 = int(0.1 * len(ids_q))
        ids_p = ids_q[:split2]
        ids_cln = ids_q[split2:]
        
        with open(foloder+'/cifar10_indices.pkl', 'wb') as f:
            pickle.dump({'ids_root': ids_root, 'ids_q': ids_q, 'ids_p': ids_p, 'ids_cln': ids_cln}, f)
    else:
        with open(foloder+'/cifar10_indices.pkl', 'rb') as f:
            dict_ids = pickle.load(f)
            ids_root = dict_ids['ids_root']; ids_q = dict_ids['ids_q']
            ids_p = dict_ids['ids_p']; ids_cln = dict_ids['ids_cln']
    # ---------------------------------------------- ed: Get the indices

    return trainset, testset, ids_root, ids_q, ids_p, ids_cln 

def prepare_CIFAR10_datasets_4(foloder, load=False, seed=42):
    ''' collect randomly 10% data as secure data
        params: 
            seed: ensure that [random split] return the same split of 
            clean and questioned every time you call this function
        return:
            trainset, testset, ids_root, ids_q, ids_p, ids_cln
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
    
    # ---------------------------------------------- st: Get the indices: (root 10%, q 90%->[10% p, 90% cln])
    if load==False:
        num_train = len(trainset)
        # Shuffle the indices
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(num_train).tolist()

        # Split indices into two subsets: 10% and 90%
        split = int(0.1 * num_train)
        ids_root = indices[:split]
        ids_q = indices[split:]

        # spilit questioned into poisoned 10%, and clean 90%
        split2 = int(0.08 * len(ids_q))
        ids_p = ids_q[:split2]
        ids_cln = ids_q[split2:]
        
        with open(foloder+'/cifar10_indices.pkl', 'wb') as f:
            pickle.dump({'ids_root': ids_root, 'ids_q': ids_q, 'ids_p': ids_p, 'ids_cln': ids_cln}, f)
    else:
        with open(foloder+'/cifar10_indices.pkl', 'rb') as f:
            dict_ids = pickle.load(f)
            ids_root = dict_ids['ids_root']; ids_q = dict_ids['ids_q']
            ids_p = dict_ids['ids_p']; ids_cln = dict_ids['ids_cln']
    # ---------------------------------------------- ed: Get the indices

    return trainset, testset, ids_root, ids_q, ids_p, ids_cln 


def prepare_CIFAR10_datasets_3(foloder, load=False, seed=42):
    ''' collect randomly 10% data as secure data, 10% as poison data, 20% as poison+noise data
        params: 
            seed: ensure that [random split] return the same split of 
            clean and questioned every time you call this function
        return:
            trainset, testset, ids_root, ids_q, ids_p, ids_cln
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
    
    # ---------------------------------------------- st: Get the indices: (root 10%, q 90%->[10% p, 90% cln])
    if load==False:
        num_train = len(trainset)
        # Shuffle the indices
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(num_train).tolist()

        # Split indices into two subsets: 10% and 90%
        split = int(0.1 * num_train)
        ids_root = indices[:split]
        ids_q = indices[split:]

        # spilit questioned into poisoned 10%, and clean 90%
        split2 = int(0.2 * len(ids_q))
        ids_p = ids_q[:split2]
        ids_cln = ids_q[split2:]
        split3 = int(0.2*len(ids_q))
        ids_noise = ids_q[split2: split2+split3]
        
        with open(foloder+'/cifar10_indices.pkl', 'wb') as f:
            pickle.dump({'ids_root': ids_root, 'ids_q': ids_q, 'ids_p': ids_p, 'ids_cln': ids_cln, 'ids_noise': ids_noise}, f)
    else:
        with open(foloder+'/cifar10_indices.pkl', 'rb') as f:
            dict_ids = pickle.load(f)
            ids_root = dict_ids['ids_root']; ids_q = dict_ids['ids_q']
            ids_p = dict_ids['ids_p']; ids_cln = dict_ids['ids_cln']
            ids_noise = dict_ids['ids_noise']
    # ---------------------------------------------- ed: Get the indices

    return trainset, testset, ids_root, ids_q, ids_p, ids_cln, ids_noise 

def prepare_CIFAR10_datasets_batt(foloder, load=False, seed=42):
    ''' collect randomly 10% data as secure data
        params: 
            seed: ensure that [random split] return the same split of 
            clean and questioned every time you call this function
        return:
            trainset, testset, ids_root, ids_q, ids_p, ids_cln
    '''
    torch.manual_seed(seed)
    transform_train = Compose([
        ToTensor(),
    ])
    transform_test = Compose([
        ToTensor(),
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
    
    # ---------------------------------------------- st: Get the indices: (root 10%, q 90%->[10% p, 90% cln])
    if load==False:
        num_train = len(trainset)
        # Shuffle the indices
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(num_train).tolist()

        # Split indices into two subsets: 10% and 90%
        split = int(0.1 * num_train)
        ids_root = indices[:split]
        ids_q = indices[split:]

        # spilit questioned into poisoned 10%, and clean 90%
        split2 = int(0.1 * len(ids_q))
        ids_p = ids_q[:split2]
        ids_cln = ids_q[split2:]
        
        with open(foloder+'/cifar10_indices.pkl', 'wb') as f:
            pickle.dump({'ids_root': ids_root, 'ids_q': ids_q, 'ids_p': ids_p, 'ids_cln': ids_cln}, f)
    else:
        with open(foloder+'/cifar10_indices.pkl', 'rb') as f:
            dict_ids = pickle.load(f)
            ids_root = dict_ids['ids_root']; ids_q = dict_ids['ids_q']
            ids_p = dict_ids['ids_p']; ids_cln = dict_ids['ids_cln']
    # ---------------------------------------------- ed: Get the indices

    return trainset, testset, ids_root, ids_q, ids_p, ids_cln 


def unnormalize(img, mean, std):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)  # inverse normalization
    return img

def predict_labels(model, images):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    return predicted

def predict_logits(model, images):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(images)
    return outputs

def truncate_logits(logits):
    truncated_logits = []
    for logit in logits:
        truncated_logit = ', '.join(f'{val:.2f}' for val in logit)
        truncated_logits.append(truncated_logit)
    return truncated_logits

def prepare_CIFAR10_datasets_SIG(foloder, target_label, load=False, seed=42):
    ''' collect randomly 10% data as secure data
        params: 
            seed: ensure that [random split] return the same split of 
            clean and questioned every time you call this function
        return:
            trainset, testset, ids_root, ids_q, ids_p, ids_cln
    '''
    torch.manual_seed(seed)
    transform_train = Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop((32, 32), padding=5),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        ToTensor(),
    ])
    transform_test = Compose([
        transforms.Resize((32, 32)),
        ToTensor(),
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
    
    # ---------------------------------------------- st: Get the indices: (root 10%, q 90%->[10% p, 90% cln])
    if load==False:
        num_train = len(trainset)
        # Shuffle the indices
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(num_train).tolist()

        # Split indices into two subsets: 10% and 90%
        split = int(0.05 * num_train)
        ids_root = indices[:split]
        ids_q = indices[split:]

        # spilit questioned into poisoned 10%, and clean 90%
        target_indices = [idx for idx in ids_q if trainset.targets[idx] == target_label]
        num_samples = int(0.1 * len(target_indices))
        ids_p = np.random.choice(target_indices, num_samples, replace=False)
        ids_cln = [idx for idx in ids_q if idx not in ids_p] 
        
        with open(foloder+'/cifar10_indices.pkl', 'wb') as f:
            pickle.dump({'ids_root': ids_root, 'ids_q': ids_q, 'ids_p': ids_p, 'ids_cln': ids_cln}, f)
    else:
        with open(foloder+'/cifar10_indices.pkl', 'rb') as f:
            dict_ids = pickle.load(f)
            ids_root = dict_ids['ids_root']; ids_q = dict_ids['ids_q']
            ids_p = dict_ids['ids_p']; ids_cln = dict_ids['ids_cln']
    # ---------------------------------------------- ed: Get the indices

    return trainset, testset, ids_root, ids_q, ids_p, ids_cln 

def prepare_CIFAR10_datasets_SIG_non_trans(foloder, target_label, load=False, seed=42):
    ''' collect randomly 10% data as secure data
        params: 
            seed: ensure that [random split] return the same split of 
            clean and questioned every time you call this function
        return:
            trainset, testset, ids_root, ids_q, ids_p, ids_cln
    '''
    torch.manual_seed(seed)
    transform_train = Compose([
        transforms.Resize((32, 32)),
        ToTensor(),
    ])
    transform_test = Compose([
        transforms.Resize((32, 32)),
        ToTensor(),
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
    
    # ---------------------------------------------- st: Get the indices: (root 10%, q 90%->[10% p, 90% cln])
    if load==False:
        num_train = len(trainset)
        # Shuffle the indices
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(num_train).tolist()

        # Split indices into two subsets: 10% and 90%
        split = int(0.05 * num_train)
        ids_root = indices[:split]
        ids_q = indices[split:]

        # spilit questioned into poisoned 10%, and clean 90%
        target_indices = [idx for idx in ids_q if trainset.targets[idx] == target_label]
        num_samples = int(0.1 * len(target_indices))
        ids_p = np.random.choice(target_indices, num_samples, replace=False)
        ids_cln = [idx for idx in ids_q if idx not in ids_p] 
        
        with open(foloder+'/cifar10_indices.pkl', 'wb') as f:
            pickle.dump({'ids_root': ids_root, 'ids_q': ids_q, 'ids_p': ids_p, 'ids_cln': ids_cln}, f)
    else:
        with open(foloder+'/cifar10_indices.pkl', 'rb') as f:
            dict_ids = pickle.load(f)
            ids_root = dict_ids['ids_root']; ids_q = dict_ids['ids_q']
            ids_p = dict_ids['ids_p']; ids_cln = dict_ids['ids_cln']
    # ---------------------------------------------- ed: Get the indices

    return trainset, testset, ids_root, ids_q, ids_p, ids_cln 

# --------------------------------- TinyImagement -------------------------------------------------
def prepare_ImageNet_datasets(foloder, load=False, seed=42):
    ''' collect randomly 10% data as secure data
        params: 
            seed: ensure that [random split] return the same split of 
            clean and questioned every time you call this function
        return:
            trainset, testset, ids_root, ids_q, ids_p, ids_cln
    '''
    torch.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    print("Loading training data")
    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    ds_tr = tiny_imagenet_dataset.TinyImageNet('../data', split='train', download=True, transform=train_transform)
    print("Loading validation data")
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    ds_te = tiny_imagenet_dataset.TinyImageNet('../data', split='val', download=False, transform=val_transform)
 
    
    # ---------------------------------------------- st: Get the indices: (root 10%, q 90%->[10% p, 90% cln])
    if load==False:
        num_train = len(ds_tr)
        # Shuffle the indices
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(num_train).tolist()

        # make 10% poisoned  
        split = int(0.1 * num_train)
        ids_p = indices[:split]
        ids_cln_all = indices[split:]

        # make 5% root  
        split2 = int(0.05 *num_train) 
        ids_root = ids_cln_all[:split2]
        ids_cln = ids_cln_all[split2:]
        ids_q = ids_p+ids_cln
        
        with open(foloder+'/cifar10_indices.pkl', 'wb') as f:
            pickle.dump({'ids_root': ids_root, 'ids_q': ids_q, 'ids_p': ids_p, 'ids_cln': ids_cln}, f)
    else:
        with open(foloder+'/cifar10_indices.pkl', 'rb') as f:
            dict_ids = pickle.load(f)
            ids_root = dict_ids['ids_root']; ids_q = dict_ids['ids_q']
            ids_p = dict_ids['ids_p']; ids_cln = dict_ids['ids_cln']
    # ---------------------------------------------- ed: Get the indices

    return ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln 
    
def prepare_ImageNet_datasets_WaNet(foloder, load=False, seed=42):
    ''' collect randomly 10% data as secure data
        params: 
            seed: ensure that [random split] return the same split of 
            clean and questioned every time you call this function
        return:
            trainset, testset, ids_root, ids_q, ids_p, ids_cln
    '''
    torch.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    print("Loading training data")
    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    ds_tr = tiny_imagenet_dataset.TinyImageNet('../data', split='train', download=True, transform=train_transform)
    print("Loading validation data")
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    ds_te = tiny_imagenet_dataset.TinyImageNet('../data', split='val', download=False, transform=val_transform)
 
    
    # ---------------------------------------------- st: Get the indices: (root 10%, q 90%->[10% p, 90% cln])
    if load==False:
        num_train = len(ds_tr)
        # Shuffle the indices
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(num_train).tolist()

        # make 10% poisoned  
        split = int(0.1 * num_train)
        ids_p = indices[:split]
        ids_noise = indices[split: int(split*2)]
        ids_cln_all = indices[int(split*2):]

        # make 5% root  
        split2 = int(0.05 *num_train) 
        ids_root = ids_cln_all[:split2]
        ids_cln = ids_cln_all[split2:]
        ids_q = ids_p+ids_cln+ids_noise
        
        with open(foloder+'/cifar10_indices.pkl', 'wb') as f:
            pickle.dump({'ids_root': ids_root, 'ids_q': ids_q, 'ids_p': ids_p, 'ids_cln': ids_cln, 'ids_noise': ids_noise}, f)
    else:
        with open(foloder+'/cifar10_indices.pkl', 'rb') as f:
            dict_ids = pickle.load(f)
            ids_root = dict_ids['ids_root']; ids_q = dict_ids['ids_q']
            ids_p = dict_ids['ids_p']; ids_cln = dict_ids['ids_cln']
            ids_noise = dict_ids['ids_noise']
    # ---------------------------------------------- ed: Get the indices

    return ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln, ids_noise 


def prepare_ImageNet_datasets_batt(foloder, load=False, seed=42):
    ''' collect randomly 10% data as secure data
        params: 
            seed: ensure that [random split] return the same split of 
            clean and questioned every time you call this function
        return:
            trainset, testset, ids_root, ids_q, ids_p, ids_cln
    '''
    torch.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    print("Loading training data")
    train_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    ds_tr = tiny_imagenet_dataset.TinyImageNet('../data', split='train', download=True, transform=train_transform)
    print("Loading validation data")
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds_te = tiny_imagenet_dataset.TinyImageNet('../data', split='val', download=False, transform=val_transform)
 
    
    # ---------------------------------------------- st: Get the indices: (root 10%, q 90%->[10% p, 90% cln])
    if load==False:
        num_train = len(ds_tr)
        # Shuffle the indices
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(num_train).tolist()

        # make 10% poisoned  
        split = int(0.1 * num_train)
        ids_p = indices[:split]
        ids_cln_all = indices[split:]

        # make 5% root  
        split2 = int(0.05 *num_train) 
        ids_root = ids_cln_all[:split2]
        ids_cln = ids_cln_all[split2:]
        ids_q = ids_p+ids_cln
        
        with open(foloder+'/cifar10_indices.pkl', 'wb') as f:
            pickle.dump({'ids_root': ids_root, 'ids_q': ids_q, 'ids_p': ids_p, 'ids_cln': ids_cln}, f)
    else:
        with open(foloder+'/cifar10_indices.pkl', 'rb') as f:
            dict_ids = pickle.load(f)
            ids_root = dict_ids['ids_root']; ids_q = dict_ids['ids_q']
            ids_p = dict_ids['ids_p']; ids_cln = dict_ids['ids_cln']
    # ---------------------------------------------- ed: Get the indices

    return ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln 

 
def prepare_ImageNet_datasets_SIG(foloder, load=False, seed=42):
    ''' collect randomly 10% data as secure data
        params: 
            seed: ensure that [random split] return the same split of 
            clean and questioned every time you call this function
        return:
            trainset, testset, ids_root, ids_q, ids_p, ids_cln
    '''
    torch.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    set_seed(42)
    print("Loading training data")
    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    ds_tr = tiny_imagenet_dataset.TinyImageNet('../data', split='train', download=True, transform=train_transform)
    print("Loading validation data")
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    ds_te = tiny_imagenet_dataset.TinyImageNet('../data', split='val', download=False, transform=val_transform)
    
    # ---------------------------------------------- st: Get the indices: (root 10%, q 90%->[10% p, 90% cln])
    if load==False:
        num_train = len(ds_tr)
        # Shuffle the indices
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(num_train).tolist()

        # make 10% poisoned  
        split = int(0.1 * num_train)
        ids_p = indices[:split]
        ids_cln_all = indices[split:]

        # make 5% root  
        split2 = int(0.05 *num_train) 
        ids_root = ids_cln_all[:split2]
        ids_cln = ids_cln_all[split2:]
        ids_q = ids_p+ids_cln
        
        with open(foloder+'/cifar10_indices.pkl', 'wb') as f:
            pickle.dump({'ids_root': ids_root, 'ids_q': ids_q, 'ids_p': ids_p, 'ids_cln': ids_cln}, f)
    else:
        with open(foloder+'/cifar10_indices.pkl', 'rb') as f:
            dict_ids = pickle.load(f)
            ids_root = dict_ids['ids_root']; ids_q = dict_ids['ids_q']
            ids_p = dict_ids['ids_p']; ids_cln = dict_ids['ids_cln']
    # ---------------------------------------------- ed: Get the indices

    return ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln 

def prepare_ImageNet_datasets_SIG_2(foloder, load=False, seed=42, target_label=None):
    ''' collect randomly 10% data as secure data
        params: 
            seed: ensure that [random split] return the same split of 
            clean and questioned every time you call this function
        return:
            trainset, testset, ids_root, ids_q, ids_p, ids_cln
    '''
    torch.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    set_seed(42)
    print("Loading training data")
    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    ds_tr = tiny_imagenet_dataset.TinyImageNet('../data', split='train', download=True, transform=train_transform)
    print("Loading validation data")
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    ds_te = tiny_imagenet_dataset.TinyImageNet('../data', split='val', download=False, transform=val_transform)
    
    # ---------------------------------------------- st: Get the indices: (root 10%, q 90%->[10% p, 90% cln])
    if load==False:
        num_train = len(ds_tr)
        # Shuffle the indices
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(num_train).tolist()

        ids_p = [idx for idx in indices if ds_tr.targets[idx] == target_label]
        ids_cln_all = [idx for idx in indices if idx not in ids_p]
        len_root = int(0.05 * num_train)
        ids_root = ids_cln_all[:len_root]
        ids_cln = ids_cln_all[len_root:]
        ids_q = ids_p+ids_cln
        
        with open(foloder+'/cifar10_indices.pkl', 'wb') as f:
            pickle.dump({'ids_root': ids_root, 'ids_q': ids_q, 'ids_p': ids_p, 'ids_cln': ids_cln}, f)
    else:
        with open(foloder+'/cifar10_indices.pkl', 'rb') as f:
            dict_ids = pickle.load(f)
            ids_root = dict_ids['ids_root']; ids_q = dict_ids['ids_q']
            ids_p = dict_ids['ids_p']; ids_cln = dict_ids['ids_cln']
    # ---------------------------------------------- ed: Get the indices

    return ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln 


def prepare_ImageNet_datasets_SIG_no_TS(foloder, load=False, seed=42):
    ''' collect randomly 10% data as secure data
        params: 
            seed: ensure that [random split] return the same split of 
            clean and questioned every time you call this function
        return:
            trainset, testset, ids_root, ids_q, ids_p, ids_cln
    '''
    torch.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    print("Loading training data")
    train_transform= Compose([
        transforms.Resize((64, 64)),
        ToTensor(),
    ])
    val_transform = Compose([
        transforms.Resize((64, 64)),
        ToTensor(),
    ]) 
    ds_tr = tiny_imagenet_dataset.TinyImageNet('../data', split='train', download=True, transform=train_transform)
    print("Loading validation data")
    ds_te = tiny_imagenet_dataset.TinyImageNet('../data', split='val', download=False, transform=val_transform)
 
    
    # ---------------------------------------------- st: Get the indices: (root 10%, q 90%->[10% p, 90% cln])
    if load==False:
        num_train = len(ds_tr)
        # Shuffle the indices
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(num_train).tolist()

        # make 10% poisoned  
        split = int(0.1 * num_train)
        ids_p = indices[:split]
        ids_cln_all = indices[split:]

        # make 5% root  
        split2 = int(0.05 *num_train) 
        ids_root = ids_cln_all[:split2]
        ids_cln = ids_cln_all[split2:]
        ids_q = ids_p+ids_cln
        
        with open(foloder+'/cifar10_indices.pkl', 'wb') as f:
            pickle.dump({'ids_root': ids_root, 'ids_q': ids_q, 'ids_p': ids_p, 'ids_cln': ids_cln}, f)
    else:
        with open(foloder+'/cifar10_indices.pkl', 'rb') as f:
            dict_ids = pickle.load(f)
            ids_root = dict_ids['ids_root']; ids_q = dict_ids['ids_q']
            ids_p = dict_ids['ids_p']; ids_cln = dict_ids['ids_cln']
    # ---------------------------------------------- ed: Get the indices

    return ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln 
   