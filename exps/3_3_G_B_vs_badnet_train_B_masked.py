import torch
import torchvision
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import copy

sys.path.append('..')
import core
from myutils import utils_data, utils_attack

exp_dir = '../experiments/exp3_GB'; label_backdoor = 6; triggerY = 6; triggerX = 6
ds_tr, ds_te, ds_x_root, ds_x_root_test, ds_x_q, ds_x_q_te = utils_data.prepare_CIFAR10_datasets(folder_=exp_dir, INITIAL_RUN=False)
assert len(ds_tr)==len(ds_x_root)+len(ds_x_q), f"wrong length, {len(ds_tr)} != {len(ds_x_root)}+{len(ds_x_q)}"
print(f'X_root: {len(ds_x_root)} samples, X_questioned: {len(ds_x_q)} samples')
bs_tr = 128
dl_te = DataLoader(
    dataset= ds_te,
    batch_size=bs_tr,
    shuffle=False,
    num_workers=0, 
    drop_last=False
)

# load model
device = torch.device("cuda:0")
model = core.models.ResNet(18); model = model.to(device); model.load_state_dict(torch.load(exp_dir+'/'+f'model_{16}.pth'))
encoder = utils_attack.Encoder_mask(); encoder = encoder.to(device); encoder.load_state_dict(torch.load(exp_dir+'/'+f'encoder_badnet_2mse_2lpips_masked{200}.pth'))
model.eval(); encoder.eval()
for param in model.parameters():
    param.requires_grad = False
for param in encoder.parameters():
    param.requires_grad = False

# Plot original and triggered images in a grid
dl_te_print = DataLoader(ds_te, batch_size=8, shuffle=True)
dataiter = iter(dl_te_print); images, labels = next(dataiter)

images = images.to(device)
images_with_triggers = images.clone().to(device); images_G = images.clone().to(device); images_bd_G = images.clone().to(device)
mask = torch.ones((3, 32, 32), dtype=torch.float32)
mask[:, ::2, ::2] = 0; mask = mask.unsqueeze(0).to(device)
for i in range(len(images_with_triggers)): 
    images_with_triggers[i] = utils_attack.add_badnet_trigger(images_with_triggers[i], triggerX=triggerX, triggerY=triggerY)
    noisy_image = encoder(images_G[i].to(device), mask) 
    images_G[i] = images_G[i].to(device)* (1 - mask) + noisy_image * mask 
    noisy_image = encoder(utils_attack.add_badnet_trigger(images_bd_G[i], triggerX=triggerX, triggerY=triggerY).to(device), mask)
    images_bd_G[i] = images_bd_G[i]* (1 - mask) + noisy_image * mask

predicted_labels_orig = utils_data.predict_labels(model, images)
predicted_labels_trig = utils_data.predict_labels(model, images_with_triggers)
predicted_labels_G_label= utils_data.predict_labels(model, images_G)
predicted_labels_bd_G_label = utils_data.predict_labels(model, images_bd_G)

predicted_labels_G_logits = utils_data.predict_logits(model, images_G)
predicted_labels_bd_G_logits = utils_data.predict_logits(model, images_bd_G)

predicted_labels_G_logits = utils_data.truncate_logits(predicted_labels_G_logits)
predicted_labels_bd_G_logits = utils_data.truncate_logits(predicted_labels_bd_G_logits)

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for idx, (ax, img) in enumerate(zip(axes[0].ravel(), images)):
    img = utils_data.unnormalize(img, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ax.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
    ax.axis('off')
    ax.set_title(ds_te.classes[predicted_labels_orig[idx]])

for idx, (ax, img) in enumerate(zip(axes[1].ravel(), images_with_triggers)):
    img = utils_data.unnormalize(img, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ax.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
    ax.axis('off')
    ax.set_title(ds_te.classes[predicted_labels_trig[idx]])

for idx, (ax, img) in enumerate(zip(axes[2].ravel(), images_G)):
    img = utils_data.unnormalize(img, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ax.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
    ax.axis('off')
    ax.set_title(f'{ds_te.classes[predicted_labels_G_label[idx]]}')

for idx, (ax, img) in enumerate(zip(axes[3].ravel(), images_bd_G)):
    img = utils_data.unnormalize(img, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ax.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
    ax.axis('off')
    ax.set_title(f'{ds_te.classes[predicted_labels_bd_G_label[idx]]}')
plt.tight_layout()


# Save the figure to a PDF file
pdf_filename = exp_dir+'/'+'cifar10_images_badnet_vs_masked_encoder_step_2.pdf'
with PdfPages(pdf_filename) as pdf:
    pdf.savefig(fig)
    plt.close()

print(f"PDF saved as {pdf_filename}")

# print the ASR for masked backdoored images
model.eval()
with torch.no_grad():
    bd_num = 0; bd_correct = 0; cln_num = 0; cln_correct = 0 
    for inputs, targets in dl_te:
        inputs_bd, targets_bd = copy.deepcopy(inputs), copy.deepcopy(targets)
        for xx in range(len(inputs_bd)):
            if targets_bd[xx]!=label_backdoor:
                noisy_image = encoder(utils_attack.add_badnet_trigger(inputs=inputs_bd[xx], triggerY=triggerY, triggerX=triggerX).to(device), mask)
                inputs_bd[xx] = inputs_bd[xx].to(device)* (1 - mask) + noisy_image * mask
                targets_bd[xx] = label_backdoor
                bd_num+=1
            else:
                targets_bd[xx] = -1
        inputs_bd, targets_bd = inputs_bd.to(device), targets_bd.to(device)
        inputs, targets = inputs.to(device), targets.to(device)
        bd_log_probs = model(inputs_bd)
        bd_y_pred = bd_log_probs.data.max(1, keepdim=True)[1]
        bd_correct += bd_y_pred.eq(targets_bd.data.view_as(bd_y_pred)).long().cpu().sum()
        encoded_inputs =encoder(inputs, mask)
        noisy_image = inputs*(1-mask) + noisy_image*mask 
        log_probs = model(noisy_image)
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        cln_correct += y_pred.eq(targets.data.view_as(y_pred)).long().cpu().sum()
        cln_num += len(inputs)
    ASR = 100.00 * float(bd_correct) / bd_num 
    ACC = 100.00 * float(cln_correct) / cln_num
    print(f'masked - ASR: {ASR: .2f}, ACC: {ACC: .2f}')
