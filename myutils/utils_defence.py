import torch

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