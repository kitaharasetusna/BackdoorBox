import matplotlib.pyplot as plt
import pickle

font_size = 15
# configs:
exp_dir = '../experiments/exp4_FIM/ISSBA'

with open(exp_dir+'/FIM.pkl', 'rb') as f:
    FIM_dict = pickle.load(f)

fim_cln = FIM_dict['clean FIM']; fim_bd = FIM_dict['backdoor FIM']
loss_cln = FIM_dict['clean loss']; loss_bd = FIM_dict['backdoor loss']
fim_cln = [t.item() for t in fim_cln]; fim_bd = [t.item() for t in fim_bd]

fig, ax1 = plt.subplots(figsize=(10, 6))

x = [i for i in range(len(loss_cln))]
# Plotting on the first axis
ax1.plot(x, fim_cln, 's-', label='clean FI')
ax1.plot(x, fim_bd,  's-', label='backdoor FI')
ax1.tick_params(axis='y')

ax1.set_xlabel('Epoch', fontsize=font_size)
ax1.set_ylabel('FI', fontsize=font_size)

# Creating the second axis
ax1.plot(x, loss_cln, 'o--',  label='clean loss', color='cyan')  # Explicit color
ax1.plot(x, loss_bd,  'o--', label='backdoor loss', color='orange')  # Explicit color
ax2 = ax1.twinx()
ax2.set_ylabel('loss', fontsize=font_size)
ax2.tick_params(axis='y')

# Adding legends for both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
ax1.legend(lines_1, labels_1, fontsize=font_size)

plt.grid(True, linestyle='--')
plt.savefig('fim_ISSBA.pdf')

plt.figure(figsize=(10, 6))
plt.plot(x, loss_cln, 'o--',  label='clean loss', color='cyan')  # Explicit color
plt.plot(x, loss_bd,  'o--', label='backdoor loss', color='orange')  # Explicit color
ax1.set_xlabel('Epoch', fontsize=font_size)
ax1.set_ylabel('FI', fontsize=font_size)
plt.legend(fontsize=font_size)
plt.savefig('loss_ISSBA.pdf')