import torch
import torch.nn as nn
import pickle

exp_dir = '../experiments/exp6_FI_B/WaNet' 

# step 1 pkl
with open(exp_dir+f'/step1_train_wanet.pkl', 'rb') as f:
    dict_step1 = pickle.load(f)

print(dict_step1.keys())
ACC_ = dict_step1['ACC']; ASR_ = dict_step1['ASR']
# for i in range(len(ACC_)):
#     print(f'ACC: {ACC_[i]}, ASR: {ASR_[i]}')
index_min_acc = ACC_.index(min(ACC_))
print(f'ACC: {min(ACC_)}, ASR: {ASR_[index_min_acc]}')