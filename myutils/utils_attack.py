import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from torchvision import transforms
from PIL import Image
from torch.utils.data import Subset
import pickle
import numpy as np
import copy

def get_next_batch(loader_iter, loader):
    try:
        return next(loader_iter)
    except StopIteration:
        return next(iter(loader))

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
    return ACC, ASR

class CustomCIFAR10Badnet(torch.utils.data.Dataset):
    def __init__(self, original_dataset, subset_indices, trigger_indices, label_bd, triggerY, triggerX):
        self.original_dataset = Subset(original_dataset, subset_indices)
        self.trigger_indices = set(trigger_indices)
        self.bd_label = label_bd
        self.triggerY = triggerY
        self.triggerX = triggerX 

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_idx = self.original_dataset.indices[idx]  # Get the original index
        image, label = self.original_dataset.dataset[original_idx]
        
        if original_idx in self.trigger_indices:
            image = add_badnet_trigger(inputs=image, triggerY=self.triggerY, triggerX=self.triggerX) 
            label = self.bd_label
        return image, label

class CustomCIFAR10Badnet_whole(torch.utils.data.Dataset):
    def __init__(self, original_dataset, trigger_indices, label_bd, triggerX, triggerY):
        self.original_dataset = original_dataset 
        self.trigger_indices = set(trigger_indices)
        self.bd_label = label_bd
        self.triggerY = triggerY
        self.triggerX = triggerX

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        # image = transforms.ToTensor()(image)  # Ensure image is a tensor
        if idx in self.trigger_indices:
            image = add_badnet_trigger(inputs=image, triggerY=self.triggerY, triggerX=self.triggerX) 
            label = self.bd_label
        return image, label

def test_asr_acc_Badnet_gen(dl_te, model, label_backdoor, B, device):
    model.eval()
    with torch.no_grad():
        bd_num = 0; bd_correct = 0; cln_num = 0; cln_correct = 0 
        for inputs, targets in dl_te:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            for xx in range(len(inputs_bd)):
                if targets_bd[xx]!=label_backdoor:
                    # TODO: to B
                    inputs_bd[xx] = add_ISSBA_gen(inputs=inputs_bd[xx], 
                                                      B=B, device=device) 
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


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 8, 8]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 4, 4]
            nn.ReLU(),
        )
        # Decoder network
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # [B, 3, 32, 32]
            nn.Sigmoid()  # Assume output image pixels are in the range [0, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


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

class FixedSTN(nn.Module):
    def __init__(self, input_channels=3, device=None):
        super(FixedSTN, self).__init__()
        
        # Initialize the affine transformation matrix as learnable parameters
        # Here we initialize the matrix to an identity matrix, but it should be a 2x3 matrix
        # self.affine_matrix = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0],
        #                                                [0.0, 1.0, 0.0]], dtype=torch.float32).unsqueeze(0))  # shape: [1, 2, 3]
        self.theta = nn.Parameter(torch.zeros(1))  # Start with 0 radians (no rotation)
        self.device = device

    
    def forward(self, x):
        # Get batch size
        batch_size = x.size(0)
        
        # # Expand the affine_matrix to match the batch size
        # affine_matrix_expanded = self.affine_matrix.expand(batch_size, -1, -1) # [8, 2, 3]
        
        # # Generate affine grid using the expanded transformation matrix
        # grid = F.affine_grid(affine_matrix_expanded, x.size(), align_corners=False) # [8, 32, 32, 2]
        
        # # Apply the affine transformation to the input
        # x_transformed = F.grid_sample(x, grid, align_corners=False) # [8, 3, 32, 32]
        rotation_matrix = torch.zeros(x.size(0), 2, 3).to(self.device)
        rotation_matrix[:, 0, 0] = torch.cos(self.theta)
        rotation_matrix[:, 0, 1] = -torch.sin(self.theta)
        rotation_matrix[:, 1, 0] = torch.sin(self.theta)
        rotation_matrix[:, 1, 1] = torch.cos(self.theta)
        grid = F.affine_grid(rotation_matrix, x.size())
        x_transformed = F.grid_sample(x, grid)

        return x_transformed

class EncoderWithFixedTransformation(nn.Module):
    def __init__(self, input_channels=3, device=None):
        super(EncoderWithFixedTransformation, self).__init__()
        
        # Fixed Spatial Transformer Network for learning transformations
        self.fixed_stn = FixedSTN(input_channels, device)
        self.device = device
        
        # Encoder network: keeps the output shape the same as the input shape (32x32)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),  # Padding to preserve spatial size
            nn.ReLU(True),
            nn.Conv2d(16, input_channels, kernel_size=3, padding=1),  # Final layer output back to 3 channels
        )
    
    def forward(self, x):
        # Apply the learned fixed transformation using STN
        x_transformed = self.fixed_stn(x)
        # Pass the transformed input through the encoder
        x_encoded = self.encoder(x_transformed)
        return x_transformed, x_encoded 

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


class CustomCIFAR10ISSBA(torch.utils.data.Dataset):
    def __init__(self, original_dataset, subset_indices, trigger_indices, label_bd, secret, encoder, device):
        self.original_dataset = Subset(original_dataset, subset_indices)
        self.trigger_indices = set(trigger_indices)
        self.bd_label = label_bd
        self.secret = secret
        self.encoder = encoder
        self.device = device

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_idx = self.original_dataset.indices[idx]  # Get the original index
        image, label = self.original_dataset.dataset[original_idx]
        # image = transforms.ToTensor()(image)  # Ensure image is a tensor
        if original_idx in self.trigger_indices:
            image = add_ISSBA_trigger(image, self.secret, self.encoder, self.device).cpu()
            label = self.bd_label
        return image, label

class CustomCIFAR10ISSBA_whole(torch.utils.data.Dataset):
    def __init__(self, original_dataset, trigger_indices, label_bd, secret, encoder, device):
        self.original_dataset = original_dataset 
        self.trigger_indices = set(trigger_indices)
        self.bd_label = label_bd
        self.secret = secret
        self.encoder = encoder
        self.device = device

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        # image = transforms.ToTensor()(image)  # Ensure image is a tensor
        if idx in self.trigger_indices:
            image = add_ISSBA_trigger(image, self.secret, self.encoder, self.device).cpu()
            label = self.bd_label
        return image, label

def add_ISSBA_gen(inputs, B, device, clamp=False):
    image_input= inputs.to(device)
    encoded_image = B(image_input) 
    if clamp==True:
        encoded_image = encoded_image.clamp(0, 1)
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

def test_asr_acc_ISSBA_gen(dl_te, model, label_backdoor, B, device):
    model.eval()
    with torch.no_grad():
        bd_num = 0; bd_correct = 0; cln_num = 0; cln_correct = 0 
        for inputs, targets in dl_te:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            for xx in range(len(inputs_bd)):
                if targets_bd[xx]!=label_backdoor:
                    # TODO: to B
                    inputs_bd[xx] = add_ISSBA_gen(inputs=inputs_bd[xx], 
                                                      B=B, device=device) 
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

def test_acc(dl_te, model, device):
    model.eval()
    cln_num = 0.0; cln_correct = 0.0
    with torch.no_grad():
        for inputs, targets in dl_te:
            inputs, targets = inputs.to(device), targets.to(device)
            log_probs = model(inputs)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            cln_correct += y_pred.eq(targets.data.view_as(y_pred)).long().cpu().sum()
            cln_num += len(inputs)
    ACC = 100.00 * float(cln_correct) / cln_num
    print(f"root data acc: {ACC}")

def fine_tune_ISSBA(dl_root, model, label_backdoor, B, device, dl_te, epoch, secret, encoder, optimizer, criterion):
    '''
    fine tune with B_theta
    '''
    model.train()
    # ACC_, ASR_ = test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
    #                                     secret=secret, encoder=encoder, device=device) 
    for ep_ in range(epoch):
        for inputs, targets in dl_root:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            for xx in range(len(inputs_bd)):
                inputs_bd[xx] = add_ISSBA_gen(inputs=inputs_bd[xx], 
                                                        B=B, device=device) 
            inputs = torch.cat((inputs_bd,inputs), dim=0)
            targets = torch.cat((targets_bd, targets))
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # make a forward pass
            outputs = model(inputs)
            # calculate the loss
            loss = criterion(outputs, targets)
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {ep_+1}')
        ACC_, ASR_ = test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                        secret=secret, encoder=encoder, device=device) 
        test_acc(dl_te=dl_root, model=model, device=device)

def fine_tune_Badnet(dl_root, model, label_backdoor, B, device, dl_te, epoch, triggerX, triggerY, optimizer, criterion):
    '''
    fine tune with B_theta
    '''
    model.train()
    # ACC_, ASR_ = test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
    #                                     secret=secret, encoder=encoder, device=device) 
    for ep_ in range(epoch):
        for inputs, targets in dl_root:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            for xx in range(len(inputs_bd)):
                inputs_bd[xx] = add_ISSBA_gen(inputs=inputs_bd[xx], 
                                                        B=B, device=device) 
            inputs = torch.cat((inputs_bd,inputs), dim=0)
            targets = torch.cat((targets_bd, targets))
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # make a forward pass
            outputs = model(inputs)
            # calculate the loss
            loss = criterion(outputs, targets)
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {ep_+1}')
        ACC_, ASR_ = test_asr_acc_badnet(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                         triggerX=triggerX, triggerY=triggerY, device=device) 
        test_acc(dl_te=dl_root, model=model, device=device)

def fine_tune_Badnet2(dl_root, model, label_backdoor, B, device, dl_te, dl_sus, loader_root_iter, loader_sus_iter,epoch, triggerX, triggerY, optimizer, criterion):
    '''
    fine tune with B_theta
    # TODO: fine tune with malicious sample
    '''
    model.train()
    # ACC_, ASR_ = test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
    #                                     secret=secret, encoder=encoder, device=device) 
    for ep_ in range(epoch):
        for i in range(max(len(dl_root), len(dl_sus))):
            X_root, Y_root = get_next_batch(loader_root_iter, dl_root)
            X_sus, Y_sus = get_next_batch(loader_sus_iter, dl_sus)

            inputs_bd, targets_bd = copy.deepcopy(X_root), copy.deepcopy(Y_root)
            for xx in range(len(inputs_bd)):
                inputs_bd[xx] = add_ISSBA_gen(inputs=inputs_bd[xx], 
                                                        B=B, device=device) 
            inputs = torch.cat((inputs_bd,X_root), dim=0)
            targets = torch.cat((targets_bd, Y_root))
            inputs, targets = inputs.to(device), targets.to(device)
            X_sus, Y_sus = X_sus.to(device), Y_sus.to(device)
            optimizer.zero_grad()
            # make a forward pass
            outputs = model(inputs)
            # calculate the loss
            loss1 = criterion(outputs, targets)
            outputs2 = model(X_sus)
            loss2 = -criterion(outputs2, Y_sus)
            if epoch>=5:
                loss=loss1+0.01*loss2
            else:
                loss=loss1
            # loss=loss1
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {ep_+1}')
        ACC_, ASR_ = test_asr_acc_badnet(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                         triggerX=triggerX, triggerY=triggerY, device=device) 
        test_acc(dl_te=dl_root, model=model, device=device)
        model.train()

def fine_tune_ISSBA2(dl_root, model, label_backdoor, B, device, dl_te, epoch, secret, encoder, optimizer, criterion):
    '''
    fine tune with B_theta
    '''
    model.train()
    ACC_, ASR_ = test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                        secret=secret, encoder=encoder, device=device) 
    for ep_ in range(epoch):
        for inputs, targets in dl_root:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            for xx in range(len(inputs_bd)):
                inputs_bd[xx] = add_ISSBA_gen(inputs=inputs_bd[xx], 
                                                        B=B, device=device) 
            inputs = inputs_bd
            targets = targets_bd
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # make a forward pass
            outputs = model(inputs)
            # calculate the loss
            loss = criterion(outputs, targets)
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {ep_+1}')
        ACC_, ASR_ = test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                        secret=secret, encoder=encoder, device=device) 


def fine_tune_pure_ISSBA(dl_root, model, label_backdoor, B, device, dl_te, epoch, secret, encoder, optimizer, criterion):
    '''fine tune'''
    model.train()
    ACC_, ASR_ = test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                        secret=secret, encoder=encoder, device=device) 
    for ep_ in range(epoch):
        for inputs, targets in dl_root:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # make a forward pass
            outputs = model(inputs)
            # calculate the loss
            loss = criterion(outputs, targets)
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {ep_+1}')
        ACC_, ASR_ = test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                        secret=secret, encoder=encoder, device=device) 


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

# --------------------------------------------------------- IAD --------------------------------------------------------------
def add_IAD_trigger(inputs, M, G, device):
    inputs = inputs.to(device); inputs = inputs.unsqueeze(0) 
    patterns = G(inputs); patterns = G.normalize_pattern(patterns)
    masks_output = M.threshold(M(inputs))
    residual = (patterns - inputs) * masks_output
    bd_inputs = inputs + residual
    return bd_inputs.squeeze(0) 

def test_asr_acc_IAD(dl_te, model, label_backdoor, M, G, device):
    model.eval()
    with torch.no_grad():
        bd_num = 0; bd_correct = 0; cln_num = 0; cln_correct = 0 
        for inputs, targets in dl_te:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            for xx in range(len(inputs_bd)):
                if targets_bd[xx]!=label_backdoor:
                    inputs_bd[xx] = add_IAD_trigger(inputs=inputs_bd[xx], M=M, G=G, device=device) 
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

class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, targets):
        return torch.ones_like(targets) * self.y_target


def create_bd(
        inputs, targets, modelG, modelM,
        y_target, device
    ):
    '''
        returns:
        bd_inputs, bd_targets, patterns, masks_output, residual
    '''
    create_targets_bd = ModifyTarget(y_target)
    bd_targets = create_targets_bd(targets).to(device)
    patterns = modelG(inputs)
    patterns = modelG.normalize_pattern(patterns)
    masks_output = modelM.threshold(modelM(inputs))
    residual = (patterns - inputs) * masks_output
    bd_inputs = inputs + residual 
    return bd_inputs, bd_targets, patterns, masks_output, residual

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



# --------------------------------------------------------- WaNet --------------------------------------------------------------
def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid

def add_WaNet_trigger(inputs, identity_grid, noise_grid, noise=False, s=0.5, grid_rescale=1, noise_rescale=2):
    '''
        make grid rescale to 1.2 when using tiny-ImageNet
    '''
    # inputs (3, 32, 32); outputs: (3, 32, 32)
    # identity_grid, noise_gird: (1, 32, 32, 2)
    h = identity_grid.shape[2]
    grid =identity_grid + s * noise_grid / h
    grid = torch.clamp(grid *grid_rescale, -1, 1)
    if noise:
        ins = torch.rand(1, h, h, 2) * noise_rescale - 1  # [-1, 1]
        grid = torch.clamp(grid + ins/h, -1, 1)
    inputs = nn.functional.grid_sample(inputs.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW
    return inputs


class CustomCIFAR10WaNet(torch.utils.data.Dataset):
    def __init__(self, original_dataset, subset_indices, trigger_indices, noise_ids, label_bd, identity_grid, noise_grid):
        self.original_dataset = Subset(original_dataset, subset_indices)
        self.trigger_indices = set(trigger_indices)
        self.noise_ids = set(noise_ids)
        self.bd_label = label_bd
        self.identity_grid  = identity_grid
        self.noise_grid = noise_grid
         

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_idx = self.original_dataset.indices[idx]  # Get the original index
        image, label = self.original_dataset.dataset[original_idx]
        
        if original_idx in self.trigger_indices:
            image = add_WaNet_trigger(inputs=image, identity_grid=self.identity_grid, 
                              noise_grid=self.noise_grid)
            label = self.bd_label
        elif original_idx in self.noise_ids:
            image = add_WaNet_trigger(inputs=image, identity_grid=self.identity_grid, 
                              noise_grid=self.noise_grid, noise=True)
        return image, label

def test_asr_acc_wanet(dl_te, model, label_backdoor, identity_grid, noise_grid, device):
    model.eval()
    with torch.no_grad():
        bd_num = 0; bd_correct = 0; cln_num = 0; cln_correct = 0 
        for inputs, targets in dl_te:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            for xx in range(len(inputs_bd)):
                if targets_bd[xx]!=label_backdoor:
                    inputs_bd[xx] = add_WaNet_trigger(inputs=inputs_bd[xx], identity_grid=identity_grid,
                                                      noise_grid=noise_grid)
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

class CustomCIFAR10WaNet_whole(torch.utils.data.Dataset):
    def __init__(self, original_dataset, trigger_indices, label_bd, identity_grid, noise_grid):
        self.original_dataset = original_dataset 
        self.trigger_indices = set(trigger_indices)
        self.bd_label = label_bd
        self.identity_grid  = identity_grid
        self.noise_grid = noise_grid

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        # image = transforms.ToTensor()(image)  # Ensure image is a tensor
        if idx in self.trigger_indices:
            image = add_WaNet_trigger(inputs=image, identity_grid=self.identity_grid,
                                                      noise_grid=self.noise_grid)
            label = self.bd_label
        return image, label

def fine_tune_Wanet2(dl_root, model, label_backdoor, B, device, dl_te, dl_sus, loader_root_iter, loader_sus_iter,epoch, identity_gird,noise_grid, optimizer, criterion):
    '''
    fine tune with B_theta
    # TODO: fine tune with malicious sample
    '''
    model.train()
    # ACC_, ASR_ = test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
    #                                     secret=secret, encoder=encoder, device=device) 
    for ep_ in range(epoch):
        for i in range(max(len(dl_root), len(dl_sus))):
            X_root, Y_root = get_next_batch(loader_root_iter, dl_root)
            X_sus, Y_sus = get_next_batch(loader_sus_iter, dl_sus)

            inputs_bd, targets_bd = copy.deepcopy(X_root), copy.deepcopy(Y_root)
            for xx in range(len(inputs_bd)):
                inputs_bd[xx] = add_ISSBA_gen(inputs=inputs_bd[xx], 
                                                        B=B, device=device) 
            inputs = torch.cat((inputs_bd,X_root), dim=0)
            targets = torch.cat((targets_bd, Y_root))
            inputs, targets = inputs.to(device), targets.to(device)
            X_sus, Y_sus = X_sus.to(device), Y_sus.to(device)
            optimizer.zero_grad()
            # make a forward pass
            outputs = model(inputs)
            # calculate the loss
            loss1 = criterion(outputs, targets)
            outputs2 = model(X_sus)
            loss2 = -criterion(outputs2, Y_sus)
            if ep_ >=5:
                loss=loss1+0.01*loss2
            else:
                loss = loss1
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {ep_+1}')
        # ACC_, ASR_ = test_asr_acc_badnet(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
        #                                  triggerX=triggerX, triggerY=triggerY, device=device) 
        ACC_, ASR_ = test_asr_acc_wanet(dl_te=dl_te, model=model,
                            label_backdoor=label_backdoor,identity_grid=identity_gird, 
                            noise_grid=noise_grid, device=device) 
        test_acc(dl_te=dl_root, model=model, device=device)
        model.train()


# --------------------------------------------------------- Blended --------------------------------------------------------------
class CustomCIFAR10Blended(torch.utils.data.Dataset):
    def __init__(self, original_dataset, subset_indices, trigger_indices, label_bd, pattern, alpha=0.5):
        self.original_dataset = Subset(original_dataset, subset_indices)
        self.trigger_indices = set(trigger_indices)
        self.bd_label = label_bd
        self.pattern = pattern
        self.alpha = alpha

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_idx = self.original_dataset.indices[idx]  # Get the original index
        image, label = self.original_dataset.dataset[original_idx]
        
        if original_idx in self.trigger_indices:
            image = add_blended_trigger(inputs=image, pattern=self.pattern, alpha=self.alpha)
            label = self.bd_label
        return image, label


def add_blended_trigger(inputs, pattern, alpha):
    inputs = (1-alpha)*inputs+alpha*pattern 
    return inputs


def test_asr_acc_blended(dl_te, model, label_backdoor, pattern, device, alpha=0.2):
    model.eval()
    with torch.no_grad():
        bd_num = 0; bd_correct = 0; cln_num = 0; cln_correct = 0 
        for inputs, targets in dl_te:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            for xx in range(len(inputs_bd)):
                if targets_bd[xx]!=label_backdoor:
                    inputs_bd[xx] = add_blended_trigger(inputs=inputs_bd[xx],
                                                        pattern=pattern, alpha=alpha)
                    # add_badnet_trigger(inputs=inputs_bd[xx], triggerY=triggerY, triggerX=triggerX)
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


class CustomCIFAR10Blended_whole(torch.utils.data.Dataset):
    def __init__(self, original_dataset, trigger_indices, label_bd, pattern, alpha=0.2):
        self.original_dataset = original_dataset 
        self.trigger_indices = set(trigger_indices)
        self.bd_label = label_bd
        self.pattern=pattern 
        self.alpha = alpha

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        # image = transforms.ToTensor()(image)  # Ensure image is a tensor
        if idx in self.trigger_indices:
            image = add_blended_trigger(inputs=image, pattern=self.pattern, alpha=self.alpha)
            label = self.bd_label
        return image, label


def fine_tune_Blended2(dl_root, model, label_backdoor, B, device, dl_te, dl_sus, loader_root_iter, loader_sus_iter,epoch, pattern, optimizer, criterion, alpha=0.2):
    '''
    fine tune with B_theta
    # TODO: fine tune with malicious sample
    '''
    model.train()
    # ACC_, ASR_ = test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
    #                                     secret=secret, encoder=encoder, device=device) 
    test_asr_acc_blended(dl_te=dl_te, model=model,
                            label_backdoor=label_backdoor, pattern=pattern, device=device, alpha=alpha)
    for ep_ in range(epoch):
        for i in range(max(len(dl_root), len(dl_sus))):
            X_root, Y_root = get_next_batch(loader_root_iter, dl_root)
            X_sus, Y_sus = get_next_batch(loader_sus_iter, dl_sus)

            inputs_bd, targets_bd = copy.deepcopy(X_root), copy.deepcopy(Y_root)
            for xx in range(len(inputs_bd)):
                inputs_bd[xx] = add_ISSBA_gen(inputs=inputs_bd[xx], 
                                                        B=B, device=device, clamp=True) 
            inputs = torch.cat((inputs_bd,X_root), dim=0)
            targets = torch.cat((targets_bd, Y_root))
            inputs, targets = inputs.to(device), targets.to(device)
            X_sus, Y_sus = X_sus.to(device), Y_sus.to(device)
            optimizer.zero_grad()
            # make a forward pass
            outputs = model(inputs)
            # calculate the loss
            loss1 = criterion(outputs, targets)
            outputs2 = model(X_sus)
            loss2 = -criterion(outputs2, Y_sus)
            loss=loss1
            if ep_ >=5:
                loss=loss1+0.03*loss2
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {ep_+1}')
        # ACC_, ASR_ = test_asr_acc_badnet(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
        #                                  triggerX=triggerX, triggerY=triggerY, device=device) 
        ACC_, ASR_ = test_asr_acc_blended(dl_te=dl_te, model=model,
                            label_backdoor=label_backdoor, pattern=pattern, device=device,alpha=alpha)
        test_acc(dl_te=dl_root, model=model, device=device)
        model.train()


def fine_tune_BATT2_1(dl_root, model, label_backdoor, B, device, dl_te, dl_sus, loader_root_iter, loader_sus_iter,epoch, rotation, optimizer, criterion):
    '''
    fine tune with B_theta
    # TODO: fine tune with malicious sample
    '''
    model.train()
    # ACC_, ASR_ = test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
    #                                     secret=secret, encoder=encoder, device=device) 
    ACC_, ASR_ = test_asr_acc_batt(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                    rotation=rotation, device=device) 
    for ep_ in range(epoch):
        for i in range(max(len(dl_root), len(dl_sus))):
            X_root, Y_root = get_next_batch(loader_root_iter, dl_root)
            X_sus, Y_sus = get_next_batch(loader_sus_iter, dl_sus)

            inputs_bd, targets_bd = copy.deepcopy(X_root), copy.deepcopy(Y_root)
            for xx in range(len(inputs_bd)):
                inputs_bd[xx] = add_BATT_gen(inputs=inputs_bd[xx].unsqueeze(0), 
                                                        B=B, device=device) 
            inputs = X_root 
            targets = Y_root 
            
            inputs_bd, targets_bd = inputs_bd.to(device), targets_bd.to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            X_sus, Y_sus = X_sus.to(device), Y_sus.to(device)
            optimizer.zero_grad()
            # make a forward pass
            outputs = model(inputs)
            outputs_bd = model(inputs_bd)
            # calculate the loss
            loss1 = criterion(outputs, targets)
            outputs2 = model(X_sus)
            loss2 = -criterion(outputs2, Y_sus)
            loss_bd = criterion(outputs_bd, targets_bd)
            loss=loss1+0.03*loss_bd
            if ep_ >=5:
                loss=loss1+loss_bd+0.03*loss2
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {ep_+1}')
        # ACC_, ASR_ = test_asr_acc_badnet(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
        #                                  triggerX=triggerX, triggerY=triggerY, device=device) 
        ACC_, ASR_ = test_asr_acc_batt(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                    rotation=rotation, device=device) 
        model.train()


def fine_tune_Blended3(dl_root, model, label_backdoor, B, device, dl_te, dl_sus, loader_root_iter, loader_sus_iter,epoch, pattern, optimizer, criterion, alpha=0.2):
    '''
    fine tune with B_theta
    # TODO: fine tune with malicious sample
    '''
    model.train()
    # ACC_, ASR_ = test_asr_acc_ISSBA(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
    #                                     secret=secret, encoder=encoder, device=device) 
    ACC_, ASR_ = test_asr_acc_blended(dl_te=dl_te, model=model,
                            label_backdoor=label_backdoor, pattern=pattern, device=device, alpha=alpha)
    test_acc(dl_te=dl_root, model=model, device=device)
    for ep_ in range(epoch):
        for i in range(max(len(dl_root), len(dl_sus))):
            X_root, Y_root = get_next_batch(loader_root_iter, dl_root)
            X_sus, Y_sus = get_next_batch(loader_sus_iter, dl_sus)

            inputs_bd, targets_bd = copy.deepcopy(X_root), copy.deepcopy(Y_root)
            for xx in range(len(inputs_bd)):
                inputs_bd[xx] = add_ISSBA_gen(inputs=inputs_bd[xx], 
                                                        B=B, device=device) 
            inputs, targets = X_root.to(device), Y_root.to(device)
            inputs_bd, targets_bd = inputs_bd.to(device), targets_bd.to(device)
            X_sus, Y_sus = X_sus.to(device), Y_sus.to(device)
            optimizer.zero_grad()
            # make a forward pass
            outputs = model(inputs)
            outputs_bd = model(inputs_bd)
            # calculate the loss
            loss1 = criterion(outputs, targets)
            outputs2 = model(X_sus)
            loss2 = -criterion(outputs2, Y_sus)
            loss3 = criterion(outputs_bd, targets_bd)
            loss=0.01*loss3+loss1+0.1*loss2
            # loss=loss1+0.01*loss2
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {ep_+1}')
        # ACC_, ASR_ = test_asr_acc_badnet(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
        #                                  triggerX=triggerX, triggerY=triggerY, device=device) 
        ACC_, ASR_ = test_asr_acc_blended(dl_te=dl_te, model=model,
                            label_backdoor=label_backdoor, pattern=pattern, device=device, alpha=alpha)
        test_acc(dl_te=dl_root, model=model, device=device)
        model.train()


# --------------------------------------------------------- BATT --------------------------------------------------------------
def add_batt_trigger(inputs, rotation, denorm=False, trans_de=None):
    if denorm == True:
        inputs = trans_de(inputs)
    # Convert tensor to PIL image for rotation
    img_pil = transforms.ToPILImage()(inputs)
    # Rotate the PIL image
    img_pil = img_pil.rotate(rotation)
    # Convert back to tensor
    inputs = transforms.ToTensor()(img_pil) 
    return inputs

def add_random_batt_trigger(inputs, denorm=False, trans_de=None):
    if denorm == True:
       inputs = trans_de(inputs) 
    # Convert tensor to PIL image for rotation
    img_pil = transforms.ToPILImage()(inputs)
    # Rotate the PIL image
    img_pil = transforms.RandomAffine(degrees=10)(img_pil)
    # Convert back to tensor
    img_tensor = transforms.ToTensor()(img_pil) 
    return inputs

class CustomCIFAR10BATT(torch.utils.data.Dataset):
    def __init__(self, original_dataset, subset_indices, trigger_indices, label_bd, roation, unorm=False, denorm=None):
        self.original_dataset = Subset(original_dataset, subset_indices)
        self.trigger_indices = set(trigger_indices)
        self.bd_label = label_bd
        self.rotation = roation
        self.unorm=unorm
        self.denorm = denorm

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_idx = self.original_dataset.indices[idx]  # Get the original index
        image, label = self.original_dataset.dataset[original_idx]
        transform1 = Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.247, 0.243, 0.261))
        ])
        if original_idx in self.trigger_indices:
            # TODO: change this to BATT
            image = add_batt_trigger(inputs=image, rotation=self.rotation, denorm=self.unorm, trans_de=self.denorm) 
            #add_badnet_trigger(inputs=image, triggerY=self.triggerY, triggerX=self.triggerX) 
            label = self.bd_label
        else:
            image = add_random_batt_trigger(inputs=image, denorm=self.unorm, trans_de=self.denorm)
        return image, label


def test_asr_acc_batt(dl_te, model, label_backdoor, rotation, device):
    model.eval()
    with torch.no_grad():
        bd_num = 0; bd_correct = 0; cln_num = 0; cln_correct = 0 
        for inputs, targets in dl_te:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            for xx in range(len(inputs_bd)):
                if targets_bd[xx]!=label_backdoor:
                    inputs_bd[xx] = add_batt_trigger(inputs=inputs_bd[xx], rotation=rotation)
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

class CustomCIFAR10BATT_whole(torch.utils.data.Dataset):
    def __init__(self, original_dataset, trigger_indices, label_bd, rotation):
        self.original_dataset = original_dataset 
        self.trigger_indices = set(trigger_indices)
        self.bd_label = label_bd
        self.rotation = rotation

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        if idx in self.trigger_indices:
            image = add_batt_trigger(inputs=image, rotation=self.rotation)
            label = self.bd_label
        return image, label


def add_BATT_gen(inputs, B, device):
    image_input= inputs.to(device)
    encoded_image = B(image_input) 
    # encoded_image = encoded_image.clamp(0, 1)
    return encoded_image.squeeze(0)

def test_asr_acc_BATT_gen(dl_te, model, label_backdoor, B, device):
    model.eval()
    with torch.no_grad():
        bd_num = 0; bd_correct = 0; cln_num = 0; cln_correct = 0 
        for inputs, targets in dl_te:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            for xx in range(len(inputs_bd)):
                if targets_bd[xx]!=label_backdoor:
                    # TODO: to B
                    inputs_bd[xx] = add_BATT_gen(inputs=inputs_bd[xx].unsqueeze(0), 
                                                      B=B, device=device) 
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

def test_asr_acc_BATT_gen2(dl_te, model, label_backdoor, B, device):
    model.eval()
    with torch.no_grad():
        bd_num = 0; bd_correct = 0; cln_num = 0; cln_correct = 0 
        for inputs, targets in dl_te:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            for xx in range(len(inputs_bd)):
                if targets_bd[xx]!=label_backdoor:
                    # TODO: to B
                    inputs_bd[xx] = add_ISSBA_gen(inputs=inputs_bd[xx], 
                                                      B=B, device=device) 
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


def fine_tune_BATT(dl_root, model, label_backdoor, B, device, dl_te, dl_sus, loader_root_iter, loader_sus_iter,epoch, rotation, optimizer, criterion):
    '''
    fine tune with B_theta
    # TODO: fine tune with malicious sample
    '''
    model.train()
    for ep_ in range(epoch):
        for i in range(max(len(dl_root), len(dl_sus))):
            X_root, Y_root = get_next_batch(loader_root_iter, dl_root)
            X_sus, Y_sus = get_next_batch(loader_sus_iter, dl_sus)

            inputs_bd, targets_bd = copy.deepcopy(X_root), copy.deepcopy(Y_root)
            for xx in range(len(inputs_bd)):
                inputs_bd[xx] = add_BATT_gen(inputs=inputs_bd[xx].unsqueeze(0), 
                                                        B=B, device=device) 
            inputs = torch.cat((inputs_bd,X_root), dim=0)
            targets = torch.cat((targets_bd, Y_root))
            inputs, targets = inputs.to(device), targets.to(device)
            X_sus, Y_sus = X_sus.to(device), Y_sus.to(device)
            optimizer.zero_grad()
            # make a forward pass
            outputs = model(inputs)
            # calculate the loss
            loss1 = criterion(outputs, targets)
            outputs2 = model(X_sus)
            loss2 = -criterion(outputs2, Y_sus)
            loss=loss1
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {ep_+1}')
        ACC_, ASR_ = test_asr_acc_batt(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                    rotation=rotation, device=device)
        test_acc(dl_te=dl_root, model=model, device=device)
        model.train()


def fine_tune_BATT2(dl_root, model, label_backdoor, B, device, dl_te, dl_sus, loader_root_iter, loader_sus_iter,epoch, rotation, optimizer, criterion):
    '''
    fine tune with B_theta
    # TODO: fine tune with malicious sample
    '''
    model.train()
    for ep_ in range(epoch):
        for i in range(max(len(dl_root), len(dl_sus))):
            X_root, Y_root = get_next_batch(loader_root_iter, dl_root)
            X_sus, Y_sus = get_next_batch(loader_sus_iter, dl_sus)

            inputs_bd, targets_bd = copy.deepcopy(X_root), copy.deepcopy(Y_root)
            for xx in range(len(inputs_bd)):
                inputs_bd[xx] = add_BATT_gen(inputs=inputs_bd[xx], 
                                                        B=B, device=device) 
            inputs = torch.cat((inputs_bd,X_root), dim=0)
            targets = torch.cat((targets_bd, Y_root))
            inputs, targets = inputs.to(device), targets.to(device)
            X_sus, Y_sus = X_sus.to(device), Y_sus.to(device)
            optimizer.zero_grad()
            # make a forward pass
            outputs = model(inputs)
            # calculate the loss
            loss1 = criterion(outputs, targets)
            outputs2 = model(X_sus)
            loss2 = -criterion(outputs2, Y_sus)
            loss=loss1+0.5*loss2
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {ep_+1}')
        ACC_, ASR_ = test_asr_acc_batt(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                    rotation=rotation, device=device)
        test_acc(dl_te=dl_root, model=model, device=device)
        model.train()

# -------------------------------------------------- SIG ATTACK ------------------------------------------
def add_SIG_trigger(inputs, delta, frequency):
    # # Convert tensor to PIL image for rotation
    img = transforms.ToPILImage()(inputs*255)
    img = np.float32(img)
    pattern = np.zeros_like(img)
    m = pattern.shape[1]
    for i in range(int(img.shape[0])):
            for j in range(int(img.shape[1])):
                pattern[i, j, :] = delta * np.sin(2 * np.pi * j * frequency / m)

    img = np.uint32(img) + pattern
    img = np.uint8(np.clip(img, 0, 255))
    img = Image.fromarray(img)
    inputs = transforms.ToTensor()(img)
    return inputs 

class CustomCIFAR10SIG(torch.utils.data.Dataset):
    def __init__(self, original_dataset, subset_indices, trigger_indices, label_bd, delta, frequency):
        self.original_dataset = Subset(original_dataset, subset_indices)
        self.trigger_indices = set(trigger_indices)
        self.bd_label = label_bd
        self.delta = delta
        self.frequency = frequency

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_idx = self.original_dataset.indices[idx]  # Get the original index
        image, label = self.original_dataset.dataset[original_idx]
        transform1 = Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.247, 0.243, 0.261))
        ])
        if original_idx in self.trigger_indices:
            image = add_SIG_trigger(inputs=image, delta=self.delta, frequency=self.frequency)
            label = self.bd_label
        return image, label


def test_asr_acc_sig(dl_te, model, label_backdoor, delta, freq, device):
    transform1 = Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.247, 0.243, 0.261))
    ])
    model.eval()
    with torch.no_grad():
        bd_num = 0; bd_correct = 0; cln_num = 0; cln_correct = 0 
        for inputs, targets in dl_te:
            inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
            for xx in range(len(inputs_bd)):
                if targets_bd[xx]!=label_backdoor:
                    inputs_bd[xx] = add_SIG_trigger(inputs=inputs_bd[xx], delta=delta, frequency=freq)
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


class CustomCIFAR10SIG_whole(torch.utils.data.Dataset):
    def __init__(self, original_dataset, trigger_indices, label_bd, delta, freq):
        self.original_dataset = original_dataset 
        self.trigger_indices = set(trigger_indices)
        self.bd_label = label_bd
        self.delta = delta
        self.frequency = freq 

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        if idx in self.trigger_indices:
            image = add_SIG_trigger(inputs=image, delta=self.delta, frequency=self.frequency)
            # add_batt_trigger(inputs=image, rotation=self.rotation)
            label = self.bd_label
        return image, label


def fine_tune_SIG(dl_root, model, label_backdoor, B, device, dl_te, dl_sus, loader_root_iter, loader_sus_iter,epoch, delta, freq, optimizer, criterion):
    '''
    fine tune with B_theta
    # TODO: fine tune with malicious sample
    '''
    model.train()
    for ep_ in range(epoch):
        for i in range(max(len(dl_root), len(dl_sus))):
            X_root, Y_root = get_next_batch(loader_root_iter, dl_root)
            X_sus, Y_sus = get_next_batch(loader_sus_iter, dl_sus)

            inputs_bd, targets_bd = copy.deepcopy(X_root), copy.deepcopy(Y_root)
            for xx in range(len(inputs_bd)):
                inputs_bd[xx] = add_BATT_gen(inputs=inputs_bd[xx].unsqueeze(0), 
                                                        B=B, device=device) 
            inputs = torch.cat((inputs_bd,X_root), dim=0)
            targets = torch.cat((targets_bd, Y_root))
            inputs, targets = inputs.to(device), targets.to(device)
            X_sus, Y_sus = X_sus.to(device), Y_sus.to(device)
            optimizer.zero_grad()
            # make a forward pass
            outputs = model(inputs)
            # calculate the loss
            loss1 = criterion(outputs, targets)
            outputs2 = model(X_sus)
            loss2 = -criterion(outputs2, Y_sus)
            loss=loss1
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {ep_+1}')
        ACC_, ASR_ = test_asr_acc_sig(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                    freq=freq, delta=delta, device=device)
        test_acc(dl_te=dl_root, model=model, device=device)
        model.train()

def fine_tune_SIG2(dl_root, model, label_backdoor, B, device, dl_te, dl_sus, loader_root_iter, loader_sus_iter,epoch, delta, freq, optimizer, criterion):
    '''
    fine tune with B_theta
    # TODO: fine tune with malicious sample
    '''
    ACC_, ASR_ = test_asr_acc_sig(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                    freq=freq, delta=delta, device=device)
    model.train()
    for ep_ in range(epoch):
        for i in range(max(len(dl_root), len(dl_sus))):
            X_root, Y_root = get_next_batch(loader_root_iter, dl_root)
            inputs_bd, targets_bd = copy.deepcopy(X_root), copy.deepcopy(Y_root)
            for xx in range(len(inputs_bd)):
                inputs_bd[xx] = add_BATT_gen(inputs=inputs_bd[xx].unsqueeze(0), 
                                                        B=B, device=device)  
            inputs = X_root 
            targets = Y_root 
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_bd, targets_bd = inputs_bd.to(device), targets_bd.to(device)
            
            optimizer.zero_grad()
            # make a forward pass
            outputs = model(inputs); outputs_bd = model(inputs_bd)
            # calculate the loss
            loss1 = criterion(outputs, targets); loss2 =criterion(outputs_bd, targets_bd)
            loss=loss1+0.1*loss2
            # do a backwards pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
        print(f'epoch: {ep_+1}')
        ACC_, ASR_ = test_asr_acc_sig(dl_te=dl_te, model=model, label_backdoor=label_backdoor,
                                                    freq=freq, delta=delta, device=device)
        test_acc(dl_te=dl_root, model=model, device=device)
        model.train()
