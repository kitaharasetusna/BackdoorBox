'''
This is a motivation experiment behind our optimized-based b2b
# TODO: 1. finish fine-tuning and ISSBA training and test
'''

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
import torch.nn.functional as F
import numpy as np
import random
import time
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import copy
import lpips

import sys
sys.path.append('..')
import core
from core.attacks.IAD import Generator
from core.attacks.ISSBA import StegaStampEncoder, StegaStampDecoder, Discriminator
from core.utils import Log

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

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
    
class GetPoisonedDatasetInt(torch.utils.data.Dataset):
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
        label = torch.tensor(self.labels[index], dtype=torch.int64)
        return img, label

# ------------------------------------------------ 0 --------------------------------------------------
# configs
y_target_IAD = 1; y_target_ISSBA = 2
folder_path = '../experiments/exp1_IAD_vs_ISSBA/'
log = Log(log_path=folder_path+f'{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.txt')
os.makedirs(folder_path, exist_ok=True)
device = torch.device("cuda:0")
train_IAD = False 
# ------------------------------------------------ 1 --------------------------------------------------
# prepare secure dataset, training dataset and test dataset

def prepare_datasets():
    ''' collect randomly 10% data as secure data
        return:
            ds_tr, ds_te, ds_secure_tr, ds_secure_te, ds_bd_tr, ds_bd_te 
    '''
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

    subset_size_tr = int(len(trainset) * 0.1); remaining_size_tr = len(trainset) - subset_size_tr
    subset_tr, ds_bd_tr = random_split(trainset, [subset_size_tr, remaining_size_tr])
    subset_size_te = int(len(testset) * 0.1); remaining_size_te = len(testset) - subset_size_te 
    subset_te, ds_bd_te = random_split(testset, [subset_size_te, remaining_size_te])

    return trainset, testset, subset_tr, subset_te, ds_bd_tr, ds_bd_te

ds_tr, ds_te, ds_secure_tr, ds_secure_te, ds_bd_tr, ds_bd_te = prepare_datasets()
# print(len(ds_tr), len(ds_te), len(ds_secure_tr), len(ds_secure_te)); 50000 10000 5000 1000

# ------------------------------------------------ 2 --------------------------------------------------# 
# Initiate a model, and train it on secure dataset (ds_secure_tr) with a benign backdoor (IAD)

def get_dataloader_order(dataloader):
    order = []
    for batch in dataloader:
        inputs, targets = batch
        order.extend(targets.tolist())
    return order

def train_mask_step(
        modelM, optimizerM, schedulerM, 
        train_dl1, train_dl2, device, 
        EPSILON, lambda_div, mask_density, lambda_norm
    ):
        modelM.train()
        total_loss = 0
        criterion_div = nn.MSELoss(reduction="none")
        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):
            optimizerM.zero_grad()

            inputs1, targets1 = inputs1.to(device), targets1.to(device)
            inputs2, targets2 = inputs2.to(device), targets2.to(device)

            # Generate the mask of data
            masks1 = modelM(inputs1)
            masks1, masks2 = modelM.threshold(modelM(inputs1)), modelM.threshold(modelM(inputs2))

            # Calculating diversity loss
            distance_images = criterion_div(inputs1, inputs2)
            distance_images = torch.mean(distance_images, dim=(1, 2, 3))
            distance_images = torch.sqrt(distance_images)

            distance_patterns = criterion_div(masks1, masks2)
            distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
            distance_patterns = torch.sqrt(distance_patterns)

            loss_div = distance_images / (distance_patterns + EPSILON)
            loss_div = torch.mean(loss_div) * lambda_div

            # Calculating mask magnitude loss
            loss_norm = torch.mean(F.relu(masks1 - mask_density))

            total_loss = lambda_norm * loss_norm + lambda_div * loss_div
            total_loss.backward()
            optimizerM.step()

        schedulerM.step()
        return total_loss, loss_norm, loss_div

class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, targets):
        return torch.ones_like(targets) * self.y_target


def create_bd(
        inputs, targets, modelG, modelM,
        y_target, device
    ):
        create_targets_bd = ModifyTarget(y_target)
        bd_targets = create_targets_bd(targets).to(device)
        patterns = modelG(inputs)
        patterns = modelG.normalize_pattern(patterns)
        masks_output = modelM.threshold(modelM(inputs))
        bd_inputs = inputs + (patterns - inputs) * masks_output
        return bd_inputs, bd_targets, patterns, masks_output

def create_cross(
        inputs1, 
        inputs2, 
        modelG, 
        modelM
    ):
        """Construct the cross samples to implement the diversity loss in [1].
        
        Args:
            inputs1 (torch.Tensor): Benign samples.
            inputs2 (torch.Tensor): Benign samples different from inputs1.
            modelG (torch.nn.Module): Backdoor trigger pattern generator.
            modelM (torch.nn.Module): Backdoor trigger mask generator.
        """
        patterns2 = modelG(inputs2)
        patterns2 = modelG.normalize_pattern(patterns2)
        masks_output = modelM.threshold(modelM(inputs2))
        inputs_cross = inputs1 + (patterns2 - inputs1) * masks_output
        return inputs_cross, patterns2, masks_output


def train_step(
        model, modelG, modelM, 
        optimizerC, optimizerG, schedulerC, 
        schedulerG, train_dl1, train_dl2, 
        device, poisoned_rate, cross_rate, y_target,
        lambda_div, EPSILON, dataset_name 
    ):
        model.train()
        modelG.train()
        total = 0
        total_cross = 0
        total_bd = 0
        total_clean = 0

        total_correct_clean = 0
        total_cross_correct = 0
        total_bd_correct = 0

        # Construct the classification loss and the diversity loss
        total_loss = 0
        criterion = nn.CrossEntropyLoss() 
        criterion_div = nn.MSELoss(reduction="none")
        train_poisoned_data, train_poisoned_label = [], []
        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):
            optimizerC.zero_grad()

            inputs1, targets1 = inputs1.to(device), targets1.to(device)
            inputs2, targets2 = inputs2.to(device), targets2.to(device)

            # Construct the benign samples, backdoored samples and cross samples
            bs = inputs1.shape[0]
            num_bd = int(poisoned_rate * bs)
            num_cross = int(cross_rate * bs)

            inputs_bd, targets_bd, patterns1, masks1 = create_bd(inputs1[:num_bd], targets1[:num_bd], modelG, modelM, y_target=y_target,device=device)
            inputs_cross, patterns2, masks2 = create_cross(
                inputs1[num_bd : num_bd + num_cross], inputs2[num_bd : num_bd + num_cross], modelG, modelM
            )

            total_inputs = torch.cat((inputs_bd, inputs_cross, inputs1[num_bd + num_cross :]), 0)
            total_targets = torch.cat((targets_bd, targets1[num_bd:]), 0)

            train_poisoned_data += total_inputs.detach().cpu().numpy().tolist()
            train_poisoned_label += total_targets.detach().cpu().numpy().tolist()

            # Calculating the classification loss
            preds = model(total_inputs)
            loss_ce = criterion(preds, total_targets)

            # Calculating diversity loss
            distance_images = criterion_div(inputs1[:num_bd], inputs2[num_bd : num_bd + num_bd])
            distance_images = torch.mean(distance_images, dim=(1, 2, 3))
            distance_images = torch.sqrt(distance_images)

            distance_patterns = criterion_div(patterns1, patterns2)
            distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
            distance_patterns = torch.sqrt(distance_patterns)

            loss_div = distance_images / (distance_patterns + EPSILON)
            loss_div = torch.mean(loss_div) * lambda_div

            # Total loss
            total_loss = loss_ce + loss_div
            total_loss.backward()
            optimizerC.step()
            optimizerG.step()

            total += bs
            total_bd += num_bd
            total_cross += num_cross
            total_clean += bs - num_bd - num_cross

            # Calculating the clean accuracy
            total_correct_clean += torch.sum(
                torch.argmax(preds[num_bd + num_cross :], dim=1) == total_targets[num_bd + num_cross :]
            )
            # Calculating the diversity accuracy
            total_cross_correct += torch.sum(
                torch.argmax(preds[num_bd : num_bd + num_cross], dim=1) == total_targets[num_bd : num_bd + num_cross]
            )
            # Calculating the backdoored accuracy
            total_bd_correct += torch.sum(torch.argmax(preds[:num_bd], dim=1) == targets_bd)
            total_loss += loss_ce.detach() * bs
            avg_loss = total_loss / total

            acc_clean = total_correct_clean * 100.0 / total_clean
            acc_bd = total_bd_correct * 100.0 / total_bd
            acc_cross = total_cross_correct * 100.0 / total_cross

            # Saving images for debugging
            if batch_idx == len(train_dl1) - 2:
                images = modelG.denormalize_pattern(torch.cat((inputs1[:num_bd], inputs_bd), dim=2))
                file_name = "{}_images.png".format(dataset_name)
                file_path = file_name 
                torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)
        schedulerC.step()
        schedulerG.step()
        return avg_loss, acc_clean, acc_bd, acc_cross

def train_benign_backdoor(ds_secure_tr, ds_secure_te, y_target):
    '''
        args: model, modelM, modelG
    '''
    model = core.models.ResNet(18)
    loss=nn.CrossEntropyLoss()
    device = torch.device("cuda:0")
    bs_tr = 128; bs_te=128
    # torch.manual_seed(42)
    loader_tr = DataLoader(
        dataset=ds_secure_tr,
        batch_size=bs_tr,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    loader_te = DataLoader(
        dataset=ds_secure_te,
        batch_size=bs_te,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    # torch.manual_seed(50)
    loader_tr1 = DataLoader(
        dataset=ds_secure_tr,
        batch_size=bs_tr,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    loader_te1 = DataLoader(
        dataset=ds_secure_te,
        batch_size=bs_te,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    torch.manual_seed(global_seed)
    order_1 = get_dataloader_order(loader_tr); order_2 = get_dataloader_order(loader_tr1); 
    # print("First DataLoader order (first 100 elements):", order_1[:100]); print("Second DataLoader order (first 100 elements):", order_2[:100])
    assert order_1!=order_2, "ds_tr for training diversity loss failed to have different order" 
    # order_1 = get_dataloader_order(loader_te); order_2 = get_dataloader_order(loader_te1); print("First DataLoader order (first 100 elements):", order_1[:100]); print("Second DataLoader order (first 100 elements):", order_2[:100])
    
    model = model.to(device)
    model.train() 
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300, 400], 0.1)

    dataset_name = 'cifar10'; y_target=y_target; poisoned_rate=0.1; cross_rate=0.1; lambda_div=1
    lambda_norm=100; mask_density=0.032; EPSILON=1e-7; epoch_M = 10; epoch_G = 25

    # Prepare the backdoor trigger pattern generator
    modelG = Generator(dataset_name).to(device)
    optimizerG = torch.optim.Adam(modelG.parameters(), lr=0.01, betas=(0.5, 0.9))
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, [200, 300, 400, 500], 0.1)

    # Prepare the backdoor trigger mask generator
    modelM = Generator(dataset_name, out_channels=1).to(device)
    optimizerM = torch.optim.Adam(modelM.parameters(), lr=0.01, betas=(0.5, 0.9))
    schedulerM = torch.optim.lr_scheduler.MultiStepLR(optimizerM, [10, 20], 0.1)

    
    epoch = 1
    if epoch == 1:
        modelM.train()
        for i in range(epoch_M): 
            total_loss, loss_norm, loss_div = train_mask_step(modelM, optimizerM, schedulerM, loader_tr, loader_tr1, device=device,
                                             EPSILON=EPSILON, lambda_div=lambda_div, mask_density=mask_density,lambda_norm=lambda_norm)
            msg = f"epoch: {i+1} "+time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + "Train Mask loss: {:.4f} | Norm: {:.3f} | Diversity: {:.3f}\n".format(total_loss, loss_norm, loss_div)
            log(msg)
            # TODO: add test  
    modelM.eval()
    modelM.requires_grad_(False)

    
    for i in range(epoch_G):
        avg_loss, acc_clean, acc_bd, acc_cross = train_step(
                model,modelG,modelM,optimizer,optimizerG,
                scheduler,schedulerG,loader_tr,loader_tr1,
                device=device, poisoned_rate=poisoned_rate, cross_rate=cross_rate,
                y_target=y_target, lambda_div=lambda_div, EPSILON=EPSILON, dataset_name=dataset_name 
            )

        msg = f"EPOCH: {i+1}: "+ time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) +\
            "Train CE loss: {:.4f} | BA: {:.3f} | ASR: {:.3f} | Cross Accuracy: {:3f}\n".format(
            avg_loss, acc_clean, acc_bd, acc_cross)
        log(msg)
    modelG.eval(); modelG.requires_grad_(False)
    return model, modelM, modelG

def save_IAD_model(model, modelM, modelG, save_folder):
    print(" Saving IAD model ... ")
    state_dict = {
        "model": model.state_dict(),
        "modelG": modelG.state_dict(),
        "modelM": modelM.state_dict(),
    }
    ckpt_model_filename = "IAD_ckpt_" +f'_{time.strftime("%Y-%m-%d_%H:%M:%S")}' + ".pth"
    ckpt_model_path = os.path.join(save_folder, ckpt_model_filename)
    torch.save(state_dict, ckpt_model_path)

if train_IAD== True: 
    log('Train IAD from scratch')
    model, modelM, modelG = train_benign_backdoor(ds_secure_tr=ds_secure_tr, ds_secure_te=ds_secure_te, y_target=y_target_IAD)
    save_IAD_model(model, modelM=modelM, modelG=modelG, save_folder=folder_path)
    modelM.eval(); modelM.requires_grad_(False); modelG.eval(); modelG.requires_grad_(False)
else:
    log('Load IAD'); dataset_name = 'cifar10'; device = torch.device("cuda:0")
    model = core.models.ResNet(18).to(device); modelG = Generator(dataset_name).to(device);modelM = Generator(dataset_name, out_channels=1).to(device) 
    folder_name = '../experiments/exp1_IAD_vs_ISSBA'; pth_path = folder_name+'/'+'IAD_ckpt__2024-06-30_16:46:02.pth'
    state_dict = torch.load(pth_path); state_dict_keys = state_dict.keys(); keys_ = [key for key in state_dict_keys]
    model.load_state_dict(state_dict[keys_[0]]); modelG.load_state_dict(state_dict[keys_[1]]);modelM.load_state_dict(state_dict[keys_[2]])

def test_IAD(ds_te, ds_te1, model, modelG, modelM, verbose=False):
    '''test backdoor accuracy, save backdoored images
        args:
            verbose: if True then save imgs
    '''
    asr = 0; acc = 0
    device = torch.device("cuda:0")
    model.eval()
    modelG.eval(); modelM.eval()
    total_correct_clean = 0.0; total_correct_bd = 0.0; total_correct_cross = 0.0; total = 0.0
    test_poisoned_data, test_poisoned_label = [], [] 

    bs_te = 128
    loader_te = DataLoader(
        dataset=ds_te,
        batch_size=bs_te,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    loader_te1 = DataLoader(
        dataset=ds_te1,
        batch_size=bs_te,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    
    
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(loader_te)), loader_te, loader_te1):
        with torch.no_grad():
            inputs1, targets1 = inputs1.to(device), targets1.to(device)
            inputs2, targets2 = inputs2.to(device), targets2.to(device)
            bs = inputs1.shape[0]

            # Construct the benign samples and calculate the clean accuracy
            preds_clean = model(inputs1)
            correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets1)
            total_correct_clean += correct_clean

            # Construct the backdoored samples and calculate the backdoored accuracy
            inputs_bd, targets_bd, _, _ = create_bd(inputs1, targets1, modelG, modelM, y_target=y_target_IAD, device=device)

            test_poisoned_data += inputs_bd.detach().cpu().numpy().tolist()
            test_poisoned_label += targets_bd.detach().cpu().numpy().tolist()

            preds_bd = model(inputs_bd)
            correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            total_correct_bd += correct_bd

            # Construct the cross samples and calculate the diversity accuracy
            inputs_cross, _, _ = create_cross(inputs1, inputs2, modelG, modelM)
            preds_cross = model(inputs_cross)
            correct_cross = torch.sum(torch.argmax(preds_cross, 1) == targets1)
            total_correct_cross += correct_cross

            total += bs
            avg_acc_clean = total_correct_clean * 100.0 / total
            avg_acc_cross = total_correct_cross * 100.0 / total
            avg_acc_bd = total_correct_bd * 100.0 / total
    if verbose:
        for index in [100, 200, 300, 400, 500, 600]:
            with torch.no_grad(): 
                image_, _ = ds_te[index]; image_ = image_.to(device).unsqueeze(0); image = copy.deepcopy(image_)
                image_ = modelG.denormalize_pattern(image_); image_ = image_.squeeze().cpu().detach().numpy().transpose((1, 2, 0)) ;plt.imshow(image_);plt.savefig(folder_name+f'/ori_{index}.pdf')
                patterns = modelG(image); patterns = modelG.normalize_pattern(patterns); masks_output = modelM.threshold(modelM(image))
                bd_inputs = image + (patterns - image) * masks_output; bd_inputs = modelG.denormalize_pattern(bd_inputs)
                bd_inputs = bd_inputs.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
                plt.imshow(bd_inputs)
                plt.savefig(folder_name+f'/IAD_{index}.pdf')
    return avg_acc_clean, avg_acc_bd, avg_acc_cross

def log_test (avg_acc_clean, avg_acc_bd, avg_acc_cross):
    msg = "==========Test result on benign test dataset==========\n" + \
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Accuracy: {avg_acc_clean: .2f}"
    log(msg)
    msg = "==========Test result on poisoned test dataset==========\n" + \
            time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
            f"Accuracy: {avg_acc_bd: .2f}"
    log(msg)
    msg = "==========Test result on cross test dataset==========\n" + \
            time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
            f"Accuracy: {avg_acc_cross: .2f}"
    log(msg)

avg_acc_clean, avg_acc_bd, avg_acc_cross = test_IAD(ds_te=ds_secure_te, ds_te1=ds_secure_te, 
                                                    model=model, modelG=modelG, modelM=modelM, verbose=False)
log_test(avg_acc_clean, avg_acc_bd, avg_acc_cross)


# ------------------------------------------------ 3 --------------------------------------------------# 
# copy model to 2 parts; fine tune on ISSBA frist
model_clean = copy.deepcopy(model); model_mali = copy.deepcopy(model)

# TODO: tips when comparing with BATT, use high ratio

def reset_grad(optimizer, d_optimizer):
    optimizer.zero_grad()
    d_optimizer.zero_grad()

def get_secret_acc(secret_true, secret_pred):
    """The accurate for the steganography secret.

    Args:
        secret_true (torch.Tensor): Label of the steganography secret.
        secret_pred (torch.Tensor): Prediction of the steganography secret.
    """
    secret_pred = torch.round(torch.sigmoid(secret_pred))
    correct_pred = (secret_pred.shape[0] * secret_pred.shape[1]) - torch.count_nonzero(secret_pred - secret_true)
    bit_acc = torch.sum(correct_pred) / (secret_pred.shape[0] * secret_pred.shape[1])

    return bit_acc

def get_ISSBA_data(trainset, testset, poisoned_set, secret_size, bs_tr=128, poisoned_rate=0.1, encoder=None, y_target=None):
    train_dl = DataLoader(
            trainset,
            batch_size=1,
            shuffle=False,
            num_workers=0)
    
    secret = torch.FloatTensor(np.random.binomial(1, .5, secret_size).tolist()).to(device)
    cln_train_dataset, cln_train_labset, bd_train_dataset, bd_train_labset = [], [], [], []
    for idx, (img, lab) in enumerate(train_dl):
        assert img.shape == (1, 3, 32, 32), f"Image shape mismatch at batch {idx}: {img.shape} != (3, 32, 32)"
        assert lab.shape == (1,), f"Target shape mismatch at batch {idx}: {lab.shape} != (1,)"
        if idx in poisoned_set:
            img = img.to(device)
            residual = encoder([secret, img])
            encoded_image = img + residual
            encoded_image = encoded_image.clamp(0, 1)
            # print(encoded_image.shape); import sys; sys.exit()
            bd_train_dataset.append(encoded_image.cpu().detach().tolist()[0])
            bd_train_labset.append(y_target)
        else:
            cln_train_dataset.append(img.tolist()[0])
            cln_train_labset.append(lab.tolist()[0])
            assert isinstance(lab.tolist()[0], int), f"Target type mismatch  {lab.tolist()[0]} != int"

    assert len(cln_train_dataset)==len(cln_train_labset), f"{len(cln_train_dataset)}!={len(cln_train_labset)}"

    cln_train_dl = GetPoisonedDatasetInt(cln_train_dataset, cln_train_labset)
    bd_train_dl = GetPoisonedDatasetInt(bd_train_dataset, bd_train_labset)
    
    cln_train_dl = DataLoader(
        cln_train_dl,
        batch_size=cln_bs,
        shuffle=True,
        num_workers=0)
    
    bd_train_dl = DataLoader(
        bd_train_dl,
        batch_size=bd_bs,
        shuffle=True,
        num_workers=0) 
    
    first_batch = next(iter(bd_train_dl))
    images, labels = first_batch

    # Print the shapes of the images and labels
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    # Optionally, check the type and dtype of labels
    print(f"Labels type: {type(labels)}")
    print(f"Labels dtype: {labels.dtype}")
    

    return cln_train_dl

def train_model_ISSBA(model, trainset, testset, poisoned_set, epoch, secret_size, poisoned_rate, encoder, y_target):
    cln_train_dl = get_ISSBA_data(trainset=trainset, testset=testset, poisoned_set=poisoned_set,
                                  secret_size=secret_size, poisoned_rate=poisoned_rate, encoder=encoder, y_target=y_target)
    return model

def train_ISSBA(ds_bd_tr, ds_bd_te, model_mali, poison_ratio=0.1, secret_size=20, modelG=None,
                modelM=None, verbose=True, y_target=0, bs_tr = 128, lr=0.001, epoch_ = 5, enc_total_epoch = 5):
    '''
        return: 
            model_mali, encoder, decoder
    '''
    print(len(ds_bd_tr), len(ds_bd_te))
    
    # train encoder - decoder

    train_data_set = []; train_secret_set = []
    test_data_set = []; test_secret_set = []
    for idx, (img, lab) in enumerate(ds_bd_tr):
        train_data_set.append(img.tolist())
        secret = np.random.binomial(1, .5, secret_size).tolist()
        train_secret_set.append(secret)
    for idx, (img, lab) in enumerate(ds_bd_te):
        test_data_set.append(img.tolist())
        secret = np.random.binomial(1, .5, secret_size).tolist()
        test_secret_set.append(secret)
    # TODO: remove this bd_te part maybe
    train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)
    test_steg_set= GetPoisonedDataset(test_data_set, test_secret_set)

    total_num = len(ds_bd_tr); poisoned_num = int(total_num * poison_ratio)
    tmp_list = list(range(total_num)); random.shuffle(tmp_list)
    poisoned_set = frozenset(tmp_list[:poisoned_num]) 
    print(len(poisoned_set))

    encoder = StegaStampEncoder(
        secret_size=secret_size, 
        height=32, 
        width=32,
        in_channel=3).to(device)
    decoder = StegaStampDecoder(
        secret_size=secret_size, 
        height=32, 
        width=32,
        in_channel=3).to(device)
    discriminator = Discriminator(in_channel=3).to(device)

    train_dl = DataLoader(
            train_steg_set,
            batch_size=32,
            shuffle=True,
            )
    
     # defualt 20 
    enc_secret_only_epoch = 2 
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=0.0001)
    d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=0.00001)
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    for epoch in range(enc_total_epoch):
        loss_list, bit_acc_list = [], []
        for idx, (image_input, secret_input) in enumerate(train_dl):
            image_input, secret_input = image_input.to(device), secret_input.to(device)
            # print(secret_input.shape, image_input.shape)
            residual = encoder([secret_input, image_input])
            encoded_image = image_input + residual
            encoded_image = encoded_image.clamp(0, 1)
            decoded_secret = decoder(encoded_image)
            D_output_fake = discriminator(encoded_image)

            # cross entropy loss for the steganography secret
            secret_loss_op = F.binary_cross_entropy_with_logits(decoded_secret, secret_input, reduction='mean')
            
           
            lpips_loss_op = loss_fn_alex(image_input,encoded_image)
            # L2 residual regularization loss
            l2_loss = torch.square(residual).mean()
            # the critic loss calculated between the encoded image and the original image
            G_loss = D_output_fake

            if epoch < enc_secret_only_epoch:
                total_loss = secret_loss_op
            else:
                total_loss = 2.0 * l2_loss + 1.5 * lpips_loss_op.mean() + 1.5 * secret_loss_op + 0.5 * G_loss
            loss_list.append(total_loss.item())

            bit_acc = get_secret_acc(secret_input, decoded_secret)
            bit_acc_list.append(bit_acc.item())

            total_loss.backward()
            optimizer.step()
            reset_grad(optimizer, d_optimizer)
        msg = f'Epoch [{epoch + 1}] total loss: {np.mean(loss_list)}, bit acc: {np.mean(bit_acc_list)}\n'
        log(msg)
    
    encoder.eval(); decoder.eval(); 
    encoder.requires_grad_(False)
    decoder.requires_grad_(False)
    
    if verbose==True: 
        for index in [100, 200, 300, 400, 500, 600]:
            with torch.no_grad(): 
                image_, secret = test_steg_set[index];secret = secret.to(device); image_ = image_.to(device).unsqueeze(0); image = copy.deepcopy(image_)
                image_ = modelG.denormalize_pattern(image_); image_ = image_.squeeze().cpu().detach().numpy().transpose((1, 2, 0)) ;plt.imshow(image_);plt.savefig(folder_name+f'/ori_{index}.pdf')
                patterns = modelG(image); patterns = modelG.normalize_pattern(patterns); masks_output = modelM.threshold(modelM(image))
                bd_inputs = image + (patterns - image) * masks_output; bd_inputs = modelG.denormalize_pattern(bd_inputs)
                bd_inputs = bd_inputs.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
                plt.imshow(bd_inputs)
                plt.savefig(folder_name+f'/IAD_{index}.pdf')

                residual = encoder([secret, image])
                encoded_image = image+ residual
                encoded_image = modelG.denormalize_pattern(encoded_image).clamp(0, 1)
                issba_image = encoded_image.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
                
                plt.imshow(issba_image)
                plt.savefig(folder_name+f'/ISSBA_{index}.pdf')
                

    savepath = os.path.join(folder_name, 'encoder_decoder.pth')
    state = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }
    torch.save(state, savepath)
    
    # ------------------------------------- fin e-d training; training models now ----------------------------------------
    train_model_ISSBA(model=model_mali, trainset=ds_bd_tr , testset=ds_bd_te, poisoned_set=poisoned_set,
                      epoch=epoch_, secret_size=secret_size, poisoned_rate=poison_ratio, encoder=encoder, y_target=y_target)
        
    
    return model_mali, encoder, decoder

# TODO: change this back to 5 (enc_total_epoch)
model_mali, encoder, decoder = train_ISSBA(ds_bd_tr=ds_bd_tr, ds_bd_te=ds_bd_te, 
                                           model_mali=model_mali, modelG=modelG, modelM=modelM, y_target=y_target_ISSBA,
                                           enc_total_epoch=1)

torch.save(model_mali.state_dict(), folder_name+'/model_mali.pth')
avg_acc_clean, avg_acc_bd, avg_acc_cross = test_IAD(ds_te=ds_secure_te, ds_te1=ds_secure_te, 
                                                    model=model_mali, modelG=modelG, modelM=modelM, verbose=False)
log_test(avg_acc_clean, avg_acc_bd, avg_acc_cross)