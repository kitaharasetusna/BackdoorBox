import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append('..')
from myutils import utils_data
# ----------------------------------------- 0.1 configs:
exp_dir = '../experiments/exp6_FI_B/ISSBA' 
secret_size = 20; label_backdoor = 6 
bs_tr = 128
epoch_B = 30; lr_B = 1e-4; lr_ft = 1e-4

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
all_data = loss_clean+loss_mali 


# Create weights for each dataset
weights_clean = [0.5 / len(loss_clean)] * len(loss_clean)
weights_mali = [0.5 / len(loss_mali)] * len(loss_mali)
# Create histogram for both datasets with normalized counts

# Plot histograms
plt.hist(loss_clean, bins=50,  alpha=0.5, label='clean', color='blue')
plt.hist(loss_mali, bins=50,  alpha=0.5, label='mali', color='red')

# Add labels and title
plt.xlabel('Loss Value')
plt.ylabel('Count')
plt.legend(loc='upper right')


plt.savefig(exp_dir+'/step2_hist.pdf')
print('done')