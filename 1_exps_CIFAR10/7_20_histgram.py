import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import sys
import pickle
import numpy as np
import os
sys.path.append('..')
from myutils import utils_data
# ----------------------------------------- 0.1 configs:
exp_dir = '../experiments/exp6_FI_B/Badnet'
secret_size = 20; label_backdoor = 6 
bs_tr = 128
epoch_B = 30; lr_B = 1e-4; lr_ft = 1e-4
os.makedirs(exp_dir, exist_ok=True)
collect = False
if collect == False:
    ds_tr, ds_te, ids_root, ids_q, ids_p, ids_cln = utils_data.prepare_CIFAR10_datasets_2(foloder=exp_dir,
                                    load=True)

    with open(exp_dir+'/step_2_X_suspicious_dict.pkl', 'rb') as f:
        data = pickle.load(f)

    loss_clean = []
    loss_mali = []

    for index, loss in data.items():
        if index in ids_p:
            loss_mali.append(loss[1])
        else:
            loss_clean.append(loss[1])

    with open(exp_dir+'/step2_loss_mali_clean.pkl', 'wb') as f:
        pickle.dump((loss_clean, loss_mali), f)
else:
    with open(exp_dir+'/step2_loss_mali_clean.pkl', 'rb') as f:
        loss_clean, loss_mali = pickle.load(f)
        print(len(loss_clean), len(loss_mali))

# Combine the data
all_data = loss_clean + loss_mali

# Create weights for each dataset
weights_clean = [0.5 / len(loss_clean)] * len(loss_clean)
weights_mali = [0.5 / len(loss_mali)] * len(loss_mali)

# Calculate min and max of the data
data_min = min(np.min(loss_clean), np.min(loss_mali))
data_max = max(np.max(loss_clean), np.max(loss_mali))

# Set margins to 5% of the data range
margin = 0.05 * (data_max - data_min)

# Plot histograms
bins_clean = np.concatenate([np.linspace(np.min(loss_clean), 2.0, 800), np.linspace(2.0, np.max(loss_clean), 500)])
bins_mali = np.concatenate([np.linspace(np.min(loss_mali), 2.0, 800), np.linspace(2.0, np.max(loss_mali), 100)])

fig, ax = plt.subplots()  # Create main plot

# Plot clean and malicious loss histograms
ax.hist(loss_clean, bins=1000, alpha=0.5, label='clean', color='blue')
ax.hist(loss_mali, bins=1000, alpha=0.5, label='malicious', color='red')

# Set xlim to include some extra space on both sides
ax.set_xlim(data_min - margin, 5.5)

# Add labels, title, legend, and grid to the main plot
ax.set_xlabel('Loss Value')
ax.set_ylabel('Count')
ax.legend(loc='upper right')
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# --- Add an inset plot ---
ax_inset = inset_axes(ax, width="30%", height="30%", loc='center right')  # Adjust loc as needed

# Plot the histograms again for the inset, focusing on a smaller range
ax_inset.hist(loss_clean, bins=5000, alpha=0.5, label='clean', color='blue')
ax_inset.hist(loss_mali, bins=5000, alpha=0.5, label='malicious', color='red')

# Set the x and y limits for the inset (zoom in on the region of interest)
ax_inset.set_xlim(0, 0.1)  # Example zoomed-in region on the x-axis
ax_inset.set_ylim(0, 500)    # Adjust y limits based on your data

# Optionally, remove the x and y labels in the inset
ax_inset.set_xticks([])
ax_inset.set_yticks([])

# Add a connecting line between the main plot and the inset
mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

# 修改插图的虚线边框
for spine in ax_inset.spines.values():
    spine.set_linestyle('--')  # 设置虚线样式
    spine.set_edgecolor('0.5')  # 设置边框颜色

print(ax_inset)

# Save and display the plot
plt.savefig(exp_dir + '/step2_hist.pdf')
plt.show()

print('done')