import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import copy

# -------------------------------------------------------- badnet ---------------------------------------------------------------
def add_badnet_trigger(inputs, triggerY, triggerX, size=5):
    pixel_max = torch.max(inputs) if torch.max(inputs)>1 else 1
    inputs[:,triggerY:triggerY+size,
            triggerX:triggerX+size] = pixel_max
    return inputs

def test_asr_acc_badnet(dl_te, model, label_backdoor, triggerX, triggerY, device):
    model.eval()
    with torch.no_grad():
        bd_num = 0; bd_correct = 0; cln_num = 0; cln_correct = 0 
        for inputs, targets in dl_te:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            for xx in range(len(inputs_bd)):
                if targets_bd[xx]!=label_backdoor:
                    inputs_bd[xx] = add_badnet_trigger(inputs=inputs_bd[xx], triggerY=triggerY, triggerX=triggerX)
                    targets_bd[xx] = label_backdoor
                    bd_num+=1
                else:
                    targets_bd[xx] = -1
            inputs_bd, targets_bd = inputs_bd.to(device), targets_bd.to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            bd_log_probs = model(inputs_bd)
            bd_y_pred = bd_log_probs.data.max(1, keepdim=True)[1]
            bd_correct += bd_y_pred.eq(targets_bd.data.view_as(bd_y_pred)).long().cpu().sum()
            log_probs = model(inputs)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            cln_correct += y_pred.eq(targets.data.view_as(y_pred)).long().cpu().sum()
            cln_num += len(inputs)
        ASR = 100.00 * float(bd_correct) / bd_num 
        ACC = 100.00 * float(cln_correct) / cln_num
        print(f'model - ASR: {ASR: .2f}, ACC: {ACC: .2f}')

# --------------------------------------------------------- UAP  ---------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

class Encoder_no(nn.Module):
    def __init__(self):
        super(Encoder_no, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class Encoder_mask(nn.Module):
    def __init__(self):
        super(Encoder_mask, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
    
    def forward(self, x, mask):
        x = x * mask  # 通过掩码只处理部分图像
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x


def create_mask():
    mask = torch.ones((3, 32, 32), dtype=torch.float32)
    mask[:, ::2, ::2] = 0
    return mask.unsqueeze(0) 

def uniform_distribution_loss(output):
    batch_size = output.size(0)
    uniform_distribution = torch.full((batch_size, 10), 1.0 / 10).to(output.device)
    loss = F.kl_div(F.log_softmax(output, dim=1), uniform_distribution, reduction='batchmean')
    return loss

def reconstruction_loss(original, reconstructed):
    loss = F.mse_loss(reconstructed, original)
    return loss


# --------------------------------------------------------- ISSBA  ---------------------------------------------------------------
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

def get_secrets(exp_dir,ds_x_tr, ds_x_te, load=False, offline=False, secret_size=20):
    '''
        args:
            offline: if True, then only return secrest, else return both
            load: if offline is False, then use this. If True, load, else rerun
        returns: 
            train_data_set, train_secret_set, test_data_set, test_secret_set
            train_data_set, test_secret_set
    '''
    train_data_set = []; train_secret_set = []
    test_data_set = []; test_secret_set = []
    
    if offline==True:
        with open(exp_dir+'/train_secret.pkl', 'rb') as f:
            train_secret_set = pickle.load(f)
        with open(exp_dir+'/test_secret.pkl', 'rb') as f:
            test_secret_set = pickle.load(f) 
        return train_data_set, test_secret_set

    for idx, (img, lab) in enumerate(ds_x_tr):
        train_data_set.append(img.tolist())
        if load == False:
            secret = np.random.binomial(1, .5, secret_size).tolist()
            train_secret_set.append(secret)
    for idx, (img, lab) in enumerate(ds_x_te):
        test_data_set.append(img.tolist())
        if load == False:
            secret = np.random.binomial(1, .5, secret_size).tolist()
            test_secret_set.append(secret)
    if load==False:
        with open(exp_dir+'/train_secret.pkl', 'wb') as f:
            pickle.dump(train_secret_set, f)
        with open(exp_dir+'/test_secret.pkl', 'wb') as f:
            pickle.dump(test_secret_set, f)
    else:
        with open(exp_dir+'/train_secret.pkl', 'rb') as f:
            train_secret_set = pickle.load(f)
        with open(exp_dir+'/test_secret.pkl', 'rb') as f:
            test_secret_set = pickle.load(f)
     
    return train_data_set, train_secret_set, test_data_set, test_secret_set
        
def reset_grad(optimizer, d_optimizer):
    optimizer.zero_grad()
    d_optimizer.zero_grad()     

def add_ISSBA_trigger(inputs, secret, encoder, device):
    image_input, secret_input = inputs.to(device), secret.to(device)
    residual = encoder([secret_input.unsqueeze(0), image_input.unsqueeze(0)])
    encoded_image = inputs.to(device)+ residual
    # encoded_image = encoded_image.clamp(0, 1)
    return encoded_image.squeeze(0) 

def test_asr_acc_ISSBA(dl_te, model, label_backdoor, secret, encoder, device):
    model.eval()
    with torch.no_grad():
        bd_num = 0; bd_correct = 0; cln_num = 0; cln_correct = 0 
        for inputs, targets in dl_te:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            for xx in range(len(inputs_bd)):
                if targets_bd[xx]!=label_backdoor:
                    inputs_bd[xx] = add_ISSBA_trigger(inputs=inputs_bd[xx], secret=secret,
                                                      encoder=encoder, device=device) 
                    targets_bd[xx] = label_backdoor
                    bd_num+=1
                else:
                    targets_bd[xx] = -1
            inputs_bd, targets_bd = inputs_bd.to(device), targets_bd.to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            bd_log_probs = model(inputs_bd)
            bd_y_pred = bd_log_probs.data.max(1, keepdim=True)[1]
            bd_correct += bd_y_pred.eq(targets_bd.data.view_as(bd_y_pred)).long().cpu().sum()
            log_probs = model(inputs)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            cln_correct += y_pred.eq(targets.data.view_as(y_pred)).long().cpu().sum()
            cln_num += len(inputs)
        ASR = 100.00 * float(bd_correct) / bd_num 
        ACC = 100.00 * float(cln_correct) / cln_num
        print(f'model - ASR: {ASR: .2f}, ACC: {ACC: .2f}')
        return ACC, ASR

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