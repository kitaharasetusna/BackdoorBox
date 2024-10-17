import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
import torchvision.transforms as transforms
from torchcam.methods import SmoothGradCAMpp, CAM, GradCAM, LayerCAM
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize, resize, to_pil_image
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Union, List

def compute_fisher_information(model, images, labels, criterion, device='cpu', mode='sum', loss_=False):
    """
    Compute the average of the trace of the Fisher Information Matrix (FIM) for a given model and a batch of samples.

    Parameters:
    - model: The neural network model (e.g., pretrained ResNet-18).
    - images: A batch of input images.
    - labels: Corresponding labels for the input images.
    - criterion: The loss function used for calculating gradients (e.g., nn.CrossEntropyLoss).
    - device: The device to perform computation on ('cpu' or 'cuda').

    Returns:
    - avg_trace: The average of the trace of the Fisher Information Matrix.
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    images, labels = images.to(device), labels.to(device)
    images.requires_grad_(True)
    
    # Perform a forward pass through the network
    output = model(images)
    loss = criterion(output, labels)

    # Compute the gradients of the loss w.r.t. the model parameters
    model.zero_grad()
    loss.backward()

    # Compute the sum of the diagonal elements of the FIM
    fim_diagonal_sum = 0; running_loss = loss.item() 
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            grad = param.grad.data
            fim_diagonal_sum += (grad**2).sum()
            num_params += grad.numel()
    
    # Calculate the average trace value
    if mode=='sum':
        if loss_ == True:
            return fim_diagonal_sum, running_loss
        else:
            return fim_diagonal_sum 
    else:
        # TODO: add if loss==True judgement
        avg_trace = fim_diagonal_sum / num_params
        return avg_trace

def compute_fisher_information_layer_spec(model, images, labels, criterion, device='cpu', mode='sum', loss_=False):
    """
    Compute the average of the trace of the Fisher Information Matrix (FIM) for a given model and a batch of samples.

    Parameters:
    - model: The neural network model (e.g., pretrained ResNet-18).
    - images: A batch of input images.
    - labels: Corresponding labels for the input images.
    - criterion: The loss function used for calculating gradients (e.g., nn.CrossEntropyLoss).
    - device: The device to perform computation on ('cpu' or 'cuda').

    Returns:
    - avg_trace: The average of the trace of the Fisher Information Matrix.
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    images, labels = images.to(device), labels.to(device)
    images.requires_grad_(True)
    
    # Perform a forward pass through the network
    output = model(images)
    loss = criterion(output, labels)

    # Compute the gradients of the loss w.r.t. the model parameters
    model.zero_grad()
    loss.backward()

    # Compute the sum of the diagonal elements of the FIM
    fim_diagonal_sum = 0; running_loss = loss.item() 
    num_params = 0
    for param in model.linear.parameters():
        if param.requires_grad:
            grad = param.grad.data
            fim_diagonal_sum += (grad**2).sum()
            num_params += grad.numel()
    
    # Calculate the average trace value
    if mode=='sum':
        if loss_ == True:
            return fim_diagonal_sum, running_loss
        else:
            return fim_diagonal_sum 
    else:
        # TODO: add if loss==True judgement
        avg_trace = fim_diagonal_sum / num_params
        return avg_trace

def compute_loss(model, images, labels, criterion, device='cpu'):

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    images, labels = images.to(device), labels.to(device)
    images.requires_grad_(True)
    
    # Perform a forward pass through the network
    output = model(images)
    loss = criterion(output, labels)

    # Compute the gradients of the loss w.r.t. the model parameters
    model.zero_grad()
    loss.backward()

    # Compute the sum of the diagonal elements of the FIM
    fim_diagonal_sum = 0; running_loss = loss.item() 
    return running_loss 

def gaussian_kernel(x, y, sigma=0.01):
    return torch.exp(-torch.sum((x - y) ** 2) / (2 * sigma ** 2))

def mmd_loss(A, B, sigma=0.01):
    # Flatten the batches
    A_flat = A.view(A.size(0), -1)
    B_flat = B.view(B.size(0), -1)
    
    # Compute the kernel values
    xx = torch.mean(gaussian_kernel(A_flat.unsqueeze(1), A_flat.unsqueeze(0), sigma))
    yy = torch.mean(gaussian_kernel(B_flat.unsqueeze(1), B_flat.unsqueeze(0), sigma))
    xy = torch.mean(gaussian_kernel(A_flat.unsqueeze(1), B_flat.unsqueeze(0), sigma))
    
    # Compute MMD
    mmd = xx + yy - 2 * xy
    return mmd

def wasserstein_distance(A, B):
    # Flatten the batches
    A_flat = A.view(A.size(0), -1)
    B_flat = B.view(B.size(0), -1)
    
    # Compute pairwise distances
    dist = torch.cdist(A_flat, B_flat, p=1)
    
    # Compute the Wasserstein distance
    wasserstein_dist = torch.mean(dist)
    return wasserstein_dist

class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks via Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am
    
    
# ----------------------------------------- SAU -----------------------------------------
def get_dataset_normalization(dataset_name):
    # idea : given name, return the default normalization of images in the dataset
    if dataset_name == "cifar10":
        # from wanet
        dataset_normalization = (transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif dataset_name =='tiny_img':
        dataset_normalization = (transforms.Normalize([0.4802, 0.4481, 0.3975],
                                     [0.2302, 0.2265, 0.2262]))
    elif dataset_name == 'cifar100':
        '''get from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151'''
        dataset_normalization = (transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]))
    elif dataset_name == "mnist":
        dataset_normalization = (transforms.Normalize([0.5], [0.5]))
    elif dataset_name == 'tiny':
        dataset_normalization = (transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]))
    elif dataset_name == "gtsrb" or dataset_name == "celeba":
        dataset_normalization = transforms.Normalize([0, 0, 0], [1, 1, 1])
    elif dataset_name == 'imagenet':
        dataset_normalization = (
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    else:
        raise Exception("Invalid Dataset")
    return dataset_normalization

def get_dataset_denormalization(normalization: transforms.Normalize):
    mean, std = normalization.mean, normalization.std

    if mean.__len__() == 1:
        mean = - mean
    else:  # len > 1
        mean = [-i for i in mean]

    if std.__len__() == 1:
        std = 1 / std
    else:  # len > 1
        std = [1 / i for i in std]

    # copy from answer in
    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    # user: https://discuss.pytorch.org/u/svd3

    invTrans = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=std),
        transforms.Normalize(mean=mean,
                             std=[1., 1., 1.]),
    ])

    return invTrans

class Shared_PGD():
    def __init__(self, model, model_ref, beta_1 = 0.01, beta_2 = 1, norm_bound = 0.2, norm_type = 'L_inf', step_size = 0.2, num_steps = 5, init_type = 'max', loss_func = torch.nn.CrossEntropyLoss(), pert_func = None, verbose = False):
        '''
        PGD attack for generating shared adversarial examples. 
        See "Shared Adversarial Unlearning: Backdoor Mitigation by Unlearning Shared Adversarial Examples" (https://arxiv.org/pdf/2307.10562.pdf) for more details.
        Implemented by Shaokui Wei (the first author of the paper) in PyTorch.
        The code is originally implemented as a part of BackdoorBench but is not dependent on BackdoorBench, and can be used independently.
        
        args:
            model: the model to be attacked
            model_ref: the reference model to be attacked
            beta_1: the weight of adversarial loss, e.g. 0.01
            beta_2: the weight of shared loss, e.g. 1
            norm_bound: the bound of the norm of perturbation, e.g. 0.2
            norm_type: the type of norm, choose from ['L_inf', 'L1', 'L2', 'Reg']
            step_size: the step size of PGD, e.g. 0.2
            num_steps: the number of steps of PGD, e.g. 5
            init_type: the type of initialization of perturbation, choose from ['zero', 'random', 'max', 'min']
            loss_func: the loss function, e.g. nn.CrossEntropyLoss()
            pert_func: the function to process the perturbation and image, e.g. add the perturbation to image
            verbose: whether to print the information of the attack
        '''

        self.model = model
        self.model_ref = model_ref
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.norm_bound = norm_bound
        self.norm_type = norm_type
        self.step_size = step_size
        self.num_steps = num_steps
        self.init_type = init_type
        self.loss_func = loss_func
        self.verbose = verbose

        if pert_func is None:
            # simply add x to perturbation
            self.pert_func = lambda x, pert: x + pert
        else:
            self.pert_func = pert_func
            
    def projection(self, pert):
        if self.norm_type == 'L_inf':
            pert.data = torch.clamp(pert.data, -self.norm_bound , self.norm_bound)
        elif self.norm_type == 'L1':
            norm = torch.sum(torch.abs(pert), dim=(1, 2, 3), keepdim=True)
            for i in range(pert.shape[0]):
                if norm[i] > self.norm_bound:
                    pert.data[i] = pert.data[i] * self.norm_bound / norm[i].item()
        elif self.norm_type == 'L2':
            norm = torch.sum(pert ** 2, dim=(1, 2, 3), keepdim=True) ** 0.5
            for i in range(pert.shape[0]):
                if norm[i] > self.norm_bound:
                    pert.data[i] = pert.data[i] * self.norm_bound / norm[i].item()
        elif self.norm_type == 'Reg':
            pass
        else:
            raise NotImplementedError
        return pert
    
    def init_pert(self, batch_pert):
        if self.init_type=='zero':
            batch_pert.data = batch_pert.data*0
        elif self.init_type=='random':
            batch_pert.data = torch.rand_like(batch_pert.data)
        elif self.init_type=='max':
            batch_pert.data = batch_pert.data + self.norm_bound
        elif self.init_type=='min':
            batch_pert.data = batch_pert.data - self.norm_bound
        else:
            raise NotImplementedError

        return self.projection(batch_pert)

    def attack(self, images, labels, max_eps = 1, min_eps = 0):
        # Set max_eps and min_eps to valid range

        model = self.model
        model_ref = self.model_ref

        batch_pert = torch.zeros_like(images, requires_grad=True)
        batch_pert = self.init_pert(batch_pert)
        model.eval()
        model_ref.eval()

        for _ in range(self.num_steps):   
            pert_image = self.pert_func(images, batch_pert)
            ori_lab = torch.argmax(model.forward(images),axis = 1).long()
            ori_lab_ref = torch.argmax(model_ref.forward(images),axis = 1).long()

            per_logits = model.forward(pert_image)
            per_logits_ref = model_ref.forward(pert_image)

            pert_label = torch.argmax(per_logits, dim=1)
            pert_label_ref = torch.argmax(per_logits_ref, dim=1)
                
            success_attack = pert_label != ori_lab
            success_attack_ref = pert_label_ref != ori_lab_ref
            common_attack = torch.logical_and(success_attack, success_attack_ref)
            shared_attack = torch.logical_and(common_attack, pert_label == pert_label_ref)

            # Adversarial loss
            # use early stop or loss clamp to avoid very large loss
            loss_adv = torch.tensor(0.0).to(images.device)
            if torch.logical_not(success_attack).sum()!=0:
                loss_adv += F.cross_entropy(per_logits, labels, reduction='none')[torch.logical_not(success_attack)].sum()
            if torch.logical_not(success_attack_ref).sum()!=0:
                loss_adv += F.cross_entropy(per_logits_ref, labels, reduction='none')[torch.logical_not(success_attack_ref)].sum()
            loss_adv = - loss_adv/2/images.shape[0]

            # Shared loss
            # JS divergence version (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
            p_model = F.softmax(per_logits, dim=1).clamp(min=1e-8)
            p_ref = F.softmax(per_logits_ref, dim=1).clamp(min=1e-8)
            mix_p = 0.5*(p_model+p_ref)
            loss_js = 0.5*(p_model*p_model.log() + p_ref*p_ref.log()) - 0.5*(p_model*mix_p.log() + p_ref*mix_p.log())
            loss_cross = loss_js[torch.logical_not(shared_attack)].sum(dim=1).sum()/images.shape[0]

            # Update pert              
            batch_pert.grad = None
            loss_ae = self.beta_1 * loss_adv + self.beta_2 * loss_cross
            loss_ae.backward()

            batch_pert.data = batch_pert.data - self.step_size * batch_pert.grad.sign()
    
            # Projection
            batch_pert = self.projection(batch_pert)

            # Optimal: projection to S and clip to [min_eps, max_eps] to ensure the perturbation is valid. It is not necessary for backdoor defense as done in i-BAU.
            # Mannually set the min_eps and max_eps to match the dataset normalization
            # batch_pert.data = torch.clamp(batch_pert.data, min_eps, max_eps)

            if torch.logical_not(shared_attack).sum()==0:
                break
        if self.verbose:
            #print(f'Maximization End: \n Adv h: {success_attack.sum().item()}, Adv h_0: {success_attack_ref.sum().item()}, Adv Common: {common_attack.sum().item()}, Adv Share: {shared_attack.sum().item()}.\n Loss adv {loss_adv.item():.4f}, Loss share {loss_cross.item():.4f}, Loss total {loss_ae.item():.4f}.\n L1 norm: {torch.sum(batch_pert[0].abs().sum()):.4f}, L2 norm: {torch.norm(batch_pert[0]):.4f}, Linf norm: {torch.max(batch_pert[0].abs()):.4f}')                    
            pass

        return batch_pert.detach()

def grad_cam(model, image_tensor, image, class_index=None, device=None, index = 0, title_="", exp_dir=""):
    ''' return grad-cam
        Get Grad-cam of the model given an image and visualize it using over-mask
        
        Args: 
            model: a torch nn.Module instance, ususally used as a neural network
            image_tensor: a torch tensor shaped as (C, H, W) that's been regularized
            image: original image which is also a torch tensor 
            class_index: int, the index of the class you want to extract
            device: 'cpu' or 'cuda' 
            iter_: int, used to name fig when training
            index: index of the image in the dataset
            title_: str, used to name the fig when infere
        Returns:
    '''
    # Set your CAM extractor
    # with SmoothGradCAMpp(model) as cam_extractor:
    mdodel = model.eval()

    # cam = GradCAM(model, 'layer4')
    # scores = model(image_tensor.unsqueeze(0).to(config['device']))
    # grad_cam = cam(class_idx=class_index, scores=scores)
    # print(grad_cam); import sys; sys.exit()
    
    # with GradCAM(model, "layer3", input_shape=(3, 32, 32)) as cam_extractor:
    with LayerCAM(model, ["layer4"], input_shape=(3, 32, 32)) as cam_extractor:
        # Preprocess your data and feed it to the model
        print(image_tensor.unsqueeze(0).shape)
        out = model(image_tensor.unsqueeze(0).to(device)) #e.g. (3, 244, 244) -> (1, 3, 244, 244)
        # Retrieve the CAM by passing the class index and the model output
        # print(out.shape, out.squeeze(0))
        probabilities = torch.softmax(out, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        print("Predicted label:", predicted_label)
        if class_index == None:
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out) # (1, 1000)->(1000)->(1: a list of B tensors)
        else:
            activation_map = cam_extractor(class_index, out)
    # Visualize the raw CAM
    plt.imshow(activation_map[0].squeeze(0).detach().cpu().numpy()); plt.axis('off'); plt.tight_layout()
    result = overlay_mask(to_pil_image(image), to_pil_image(activation_map[0].squeeze(0).detach().cpu(), mode='F'), 
                          alpha=0.3,
                          colormap='viridis')
    # Display it
    plt.imshow(result); plt.axis('off'); plt.tight_layout()
    plt.savefig(f'{exp_dir}/{title_}_index_{index}_predicted_{predicted_label}_gt_{class_index}.pdf')


def test_f1_score(idx_sus, ids_p):
    print(f'index lenght: {len(idx_sus)}')
    TP, FP = 0.0, 0.0
    for s in idx_sus:
        if s in ids_p:
            TP+=1
        else:
            FP+=1
    print(TP/(TP+FP))


# ---------------------------------------- difussion model ------------------------------------------------------
# 
class Swish(nn.Module):
    """
    ### Swish actiavation function
    

    """

    def forward(self, x):
        return x * torch.sigmoid(x)

# The time embedding
class TimeEmbedding(nn.Module):
    """
    ### Embeddings for 
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings

        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb

# Residual blocks include 'skip' connections
class ResidualBlock(nn.Module):
    """
    ### Residual block
    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step () embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings
        h += self.time_emb(t)[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.act2(self.norm2(h)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)

# Ahh yes, magical attention...
class AttentionBlock(nn.Module):
    """
    ### Attention block
    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product 
 

        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension  	 
 

        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class DownBlock(nn.Module):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by 
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by 
 

    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)

# The core class definition (aka the important bit)
def get_beta():
    '''
        returns:
            return n_steps, beta, alpha, alpha_bar
    '''
    # Set up some parameters
    n_steps = 1000
    beta = torch.linspace(0.0001, 0.04, n_steps).cuda()
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return n_steps, beta, alpha, alpha_bar

class UNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2):
        """
        * `image_channels` is the number of channels in the image.  for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))

def tensor_to_image(t):
    ''' deprecated (no denormalization)
    tensor to PIL image (for display)
    # TODO: add denormalization
    '''
    return Image.fromarray(np.array((t.squeeze().permute(1, 2, 0))*255).astype(np.uint8))

def q_xt_xtminus1(xtm1, t, device, beta):
    ''' add noise to image xtm1
        beta: [beta_1, ..., beta_t]
    '''
    c, h, w = xtm1.shape
    xtm1 = xtm1.to(device)

    eta = torch.randn(c, h, w).to(device)

    noisy = (1 - beta[t]).sqrt().reshape(1, 1, 1) * xtm1 +\
            beta[t].sqrt().reshape(1, 1, 1) * eta
    return noisy

def q_xt_x0(x0, t, device, alpha_bar):
    ''' add noise to image xtm1
        beta: [beta_1, ..., beta_t]
    '''
    n, c, h, w = x0.shape
    x0 = x0.cuda()
    eps = torch.randn(n, c, h, w).to(device)

    a_bar = alpha_bar[t]

    noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eps
    return noisy, eps # also returns noise
    # End

def p_xt(xt, noise, t, alpha, alpha_bar, beta, device):
    ''' noise to image
    '''
    alpha_t = alpha[t]
    alpha_bar_t = alpha_bar[t]
    eps_coef = (1 - alpha_t) / (1 - alpha_bar_t) ** .5
    mean = 1 / (alpha_t ** 0.5) * (xt - eps_coef * noise)
    var = beta[t]
    eps = torch.randn(xt.shape, device=device)
    return mean + (var ** 0.5) * eps

def sampling(x_T, model, T, alpha, alpha_bar, beta, device):
    x = x_T
    for i in range(T):
        t = torch.tensor(T-i-1, dtype=torch.long).cuda()
        with torch.no_grad():
            pred_noise = model(x.float(), t.unsqueeze(0))
            x = p_xt(x, pred_noise, t.unsqueeze(0), 
                    alpha=alpha, alpha_bar=alpha_bar,
                    beta=beta, device=device)
    return x