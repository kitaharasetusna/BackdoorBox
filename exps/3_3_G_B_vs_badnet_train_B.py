import torch
import torchvision
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

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
encoder = utils_attack.Encoder(); encoder = encoder.to(device); encoder.load_state_dict(torch.load(exp_dir+'/'+f'encoder_2mse_2lpips_{200}.pth'))
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
for i in range(len(images_with_triggers)): 
    images_with_triggers[i] = utils_attack.add_badnet_trigger(images_with_triggers[i], triggerX=triggerX, triggerY=triggerY)
    images_G[i] = encoder(images_G[i].to(device)) 
    images_bd_G[i] = encoder(utils_attack.add_badnet_trigger(images_bd_G[i], triggerX=triggerX, triggerY=triggerY).to(device))

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
pdf_filename = exp_dir+'/'+'cifar10_images_2_mse_2_lpips.pdf'
with PdfPages(pdf_filename) as pdf:
    pdf.savefig(fig)
    plt.close()

print(f"PDF saved as {pdf_filename}")
