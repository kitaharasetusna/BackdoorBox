import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.distributions as dist

import sys
sys.path.append('..')
from myutils import utils_defence


exp_dir = "../experiments/diffusion_toy/"
# ------------------------------------ 1. Load CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# ------------------------------------ 2. View some examples:
image = Image.new('RGB', size=(32*5, 32*2))
# Paste 10 images onto the canvas
for i in range(10):
    im_tensor, _ = train_dataset[i] 
    im = utils_defence.tensor_to_image(im_tensor)  # Convert tensor to PIL image
    
    # Paste the image using a 2-tuple (top-left corner)
    image.paste(im, ((i%5)*32, (i//5)*32))

# Resize the final image for better visualization
image = image.resize((32*5*4, 32*2*4), Image.NEAREST)
# Show and save the image
image.save(exp_dir+"1_cifar-10_10_images.pdf", "PDF")

# ------------------------------------ 3. define device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# ------------------------------------ 4. test add nois
n_steps = 100
beta = torch.linspace(0.0001, 0.04, n_steps).to(device)

test_img, _ = train_dataset[50]
list_noisy_img = []
for t in range(n_steps):
    # Store images every 20 steps to show progression
    if t%20 == 0:
        list_noisy_img.append(utils_defence.tensor_to_image(test_img.cpu()))
    # Calculate Xt given Xt-1 (i.e. x from the previous iteration)
    t = torch.tensor(t, dtype=torch.long) # t as a tensor
    test_img = utils_defence.q_xt_xtminus1(test_img, t, device=device, beta=beta) # Modify x using our function above

image = Image.new('RGB', size=(32*5, 32))
for i, im in enumerate(list_noisy_img):
    image.paste(im, ((i%5)*32, 0))
image.resize((32*4*5, 32*4), Image.NEAREST)
image.save(exp_dir+'2_cifar-10_test_add_noise.pdf', 'PDF')

# ------------------------------------ 5. define UNet and train it

# Create the model
unet = utils_defence.UNet(n_channels=32).cuda()
n_steps, beta, alpha, alpha_bar = utils_defence.get_beta()

lr = 7e-5 # Explore this - might want it lower when training on the full dataset
losses = [] # Store losses for later plotting
optim = torch.optim.Adam(unet.parameters(), lr=lr) # Optimizer

TRAIN_UNET = False
if TRAIN_UNET:
    epochs = 30
    epoch_loss = []
    for epoch in range(1, epochs + 1):
        average_train_loss = 0
        for x0, _ in train_loader:
            optim.zero_grad()
            t = torch.randint(0, n_steps, (x0.shape[0],), dtype=torch.long).cuda()

            xt, noise = utils_defence.q_xt_x0(x0, t, device=device, alpha_bar=alpha_bar)

            pred_noise = unet(xt.float(), t)
            loss = F.mse_loss(noise.float(), pred_noise)
            losses.append(loss.item())
            average_train_loss+=loss.item()

            loss.backward()
            optim.step()

        epoch_loss.append(average_train_loss / len(train_loader))
        print(f'epoch: {epoch+1}, loss: {average_train_loss/len(train_loader)}')
        if (epoch+1)%5==0:
            torch.save(unet.state_dict(), exp_dir+f"3_diffusion_{epoch+1}.pth")
    from matplotlib import pyplot as plt
    plt.plot(epoch_loss)
    plt.savefig(exp_dir+'4_loss.pdf')
else:
    pass

x = torch.randn(1, 3, 32, 32).cuda() # Start with random noise
ims = []
model = utils_defence.UNet(n_channels=32).cuda()
model.load_state_dict(torch.load(exp_dir+'3_diffusion_30.pth', weights_only=True))

for i in range(1, n_steps+1):
  t = torch.tensor(n_steps-i , dtype=torch.long)
  t = t.cuda()
  t = t.unsqueeze(0)

  with torch.no_grad():
    predictions = model(x.float(), t)
    x = utils_defence.p_xt(x, predictions, t, 
                           alpha=alpha, alpha_bar=alpha_bar,
                           device=device, beta=beta)

    if i % 200 == 0:
      ims.append(utils_defence.tensor_to_image(x.cpu()))

image = Image.new('RGB', size=(32*5, 32))
for i, im in enumerate(ims[:5]):
  image.paste(im, ((i%5)*32, 0))
image.resize((32*4*5, 32*4), Image.NEAREST)
image.save(exp_dir+"5_generated.pdf", "PDF")

#@title Make and show 10 examples:
x = torch.randn(100, 3, 32, 32).cuda() # Start with random noise
ims = []

x = utils_defence.sampling(x_T=x, model=model,
                           T=n_steps, alpha=alpha,
                           alpha_bar=alpha_bar, beta=beta,
                           device=device)

for i in range(100):
    ims.append(utils_defence.tensor_to_image(x[i].unsqueeze(0).cpu()))

image = Image.new('RGB', size=(32*10, 32*10))
for i, im in enumerate(ims):
  image.paste(im, ((i%10)*32, 32*(i//10)))
image.resize((32*4*10, 32*4*10), Image.NEAREST)
image.save(exp_dir+"6_more_generated.pdf", "PDF")
