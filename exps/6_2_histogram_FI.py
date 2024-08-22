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

    sorted_items = sorted(data.items(), key=lambda item: item[1][0], reverse=True)
    top_10_percent_count = max(1, len(sorted_items) * 1 // 100)
    ids_suspicious = [item[0] for item in sorted_items[:top_10_percent_count]]
    TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0
    for s in ids_suspicious:
        if s in ids_p:
            TP+=1
        else:
            FP+=1
    precision = TP/(TP+FP)
    print(precision)
    loss_clean = []
    loss_mali = []

    for index, loss in data.items():
        if index in ids_p:
            loss_mali.append(loss[0])
        else:
            loss_clean.append(loss[0])

    with open(exp_dir+'/step2_FI_mali_clean.pkl', 'wb') as f:
        pickle.dump((loss_clean, loss_mali), f)
else:
    with open(exp_dir+'/step2_FI_mali_clean.pkl', 'rb') as f:
        loss_clean, loss_mali = pickle.load(f)
        print(len(loss_clean), len(loss_mali))


# Combine the data
all_data = loss_clean+loss_mali 


# Create weights for each dataset
weights_clean = [0.5 / len(loss_clean)] * len(loss_clean)
weights_mali = [0.5 / len(loss_mali)] * len(loss_mali)
# Create histogram for both datasets with normalized counts

print(max(loss_clean), max(loss_mali))
# Plot histograms
plt.hist(loss_clean, bins=30,  alpha=0.5, label='clean', color='blue')
plt.hist(loss_mali, bins=100,  alpha=0.5, label='mali', color='red')

# Add labels and title
plt.xlabel('FI Value')
plt.ylabel('Count')
plt.legend(loc='upper right')


plt.savefig(exp_dir+'/step2_hist_FI.pdf')
print('done')