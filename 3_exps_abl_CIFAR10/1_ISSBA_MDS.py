from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

exp_dir = '../experiments/exp6_FI_B/ISSBA_abl' 
A = torch.load(exp_dir+'/ndarray_mali.pth', weights_only=False)
B = torch.load(exp_dir+'/ndarray_random.pth', weights_only=False)
C = torch.load(exp_dir+'/ndarray_B_theta.pth', weights_only=False)
D  = torch.load(exp_dir+'/clean_narray.pth', weights_only=False)
C = np.array(C)

# Suppose A, B, C, D are numpy arrays with shape (640, 10)

# Step 1: Compute cosine similarities
cos_sim_A_B = cosine_similarity(A, B)
cos_sim_A_C = cosine_similarity(A, C)
cos_sim_A_D = cosine_similarity(A, D)

cos_sim_B_C = cosine_similarity(B, C)
cos_sim_B_D = cosine_similarity(B, D)

cos_sim_C_D = cosine_similarity(C, D)

cur_time = time.time()
# Step 2: Combine all cosine similarity matrices and convert to dissimilarities (1 - cosine similarity)
dissim_A_B = 1 - cos_sim_A_B
dissim_A_C = 1 - cos_sim_A_C
dissim_A_D = 1 - cos_sim_A_D

dissim_B_C = 1 - cos_sim_B_C
dissim_B_D = 1 - cos_sim_B_D

dissim_C_D = 1 - cos_sim_C_D
print("dis-sim: ", time.time()-cur_time); cur_time=time.time()
# Step 3: Build the full combined distance matrix (2560x2560)
combined_distance_matrix = np.zeros((2560, 2560))

# Fill in the dissimilarities for A, B, C, D
combined_distance_matrix[:640, 640:1280] = dissim_A_B
combined_distance_matrix[:640, 1280:1920] = dissim_A_C
combined_distance_matrix[:640, 1920:] = dissim_A_D

combined_distance_matrix[640:1280, 1280:1920] = dissim_B_C
combined_distance_matrix[640:1280, 1920:] = dissim_B_D

combined_distance_matrix[1280:1920, 1920:] = dissim_C_D
# Make the matrix symmetric
combined_distance_matrix = combined_distance_matrix + combined_distance_matrix.T
print("sim-matrix: ", time.time()-cur_time); cur_time=time.time()
# Step 4: Run MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_results = mds.fit_transform(combined_distance_matrix)
print("MSD-dim-decomp: ", time.time()-cur_time); cur_time=time.time()
# Step 5: Visualization
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 20})
plt.scatter(mds_results[:640, 0], mds_results[:640, 1], label='malicious samples', c='red', marker='x', alpha=0.5)
plt.scatter(mds_results[640:1280, 0], mds_results[640:1280, 1], 
            label='ramdomly initialized $B_\\theta$ \ngenerated samples', 
            marker='8', c='gray', alpha=0.5)
plt.scatter(mds_results[1280:1920, 0], mds_results[1280:1920, 1], label='learned $B_\\theta$\n generated samples',
            marker='*', c='orange', alpha=0.5)
plt.scatter(mds_results[1920:, 0], mds_results[1920:, 1], label='clean samples',
            marker='+', c='blue', alpha=0.5)
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')

plt.subplots_adjust(top=0.85)  # Change the value as needed (default is 0.9)
plt.grid(linestyle=':', linewidth=1)  # Dotted lines
plt.legend(frameon=False, facecolor='white',framealpha=1)

plt.savefig(exp_dir+'/sep4_MDS.pdf')
print("fig plot: ", time.time()-cur_time); cur_time=time.time()