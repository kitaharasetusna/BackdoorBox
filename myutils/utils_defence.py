import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchcam.methods import SmoothGradCAMpp, CAM, GradCAM, LayerCAM
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize, resize, to_pil_image

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