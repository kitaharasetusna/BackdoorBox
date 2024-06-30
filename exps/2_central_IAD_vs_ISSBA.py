'''
This is a motivation experiment behind our optimized-based b2b
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

import sys
sys.path.append('..')
import core
from core.attacks.IAD import Generator

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

# ------------------------------------------------ 1 --------------------------------------------------
# prepare secure dataset, training dataset and test dataset

def prepare_datasets():
    ''' collect randomly 10% data as secure data
        return:
            ds_tr, ds_te, ds_secure_tr, ds_secure_te 
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
    subset_tr, _ = random_split(trainset, [subset_size_tr, remaining_size_tr])
    subset_size_te = int(len(testset) * 0.1); remaining_size_te = len(testset) - subset_size_te 
    subset_te, _ = random_split(testset, [subset_size_te, remaining_size_te])

    return trainset, testset, subset_tr, subset_te

ds_tr, ds_te, ds_secure_tr, ds_secure_te = prepare_datasets()
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
            print(msg)
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
        print(msg)
    return model

train_benign_backdoor(ds_secure_tr=ds_secure_tr, ds_secure_te=ds_secure_te, y_target=1)

