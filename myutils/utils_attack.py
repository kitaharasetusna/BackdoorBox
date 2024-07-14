import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------- badnet ---------------------------------------------------------------
def add_badnet_trigger(inputs, triggerY, triggerX, size=5):
    pixel_max = torch.max(inputs) if torch.max(inputs)>1 else 1
    inputs[:,triggerY:triggerY+size,
            triggerX:triggerX+size] = pixel_max
    return inputs

# -------------------------------------------------------- TUAP  ---------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

def uniform_distribution_loss(output):
    batch_size = output.size(0)
    uniform_distribution = torch.full((batch_size, 10), 1.0 / 10).to(output.device)
    loss = F.kl_div(F.log_softmax(output, dim=1), uniform_distribution, reduction='batchmean')
    return loss
