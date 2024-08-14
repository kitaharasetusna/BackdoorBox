import torch
import torch.nn.functional as F

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