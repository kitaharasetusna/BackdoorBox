import matplotlib.pyplot as plt
import pickle


# configs:
exp_dir = '../experiments/exp4_FIM'

with open(exp_dir+'/FIM.pkl', 'rb') as f:
    FIM_dict = pickle.load(f)

fim_cln = FIM_dict['clean FIM']; fim_bd = FIM_dict['backdoor FIM']
fim_cln = [t.item() for t in fim_cln]; fim_bd = [t.item() for t in fim_bd]
plt.figure()
plt.plot(fim_cln, label='clean')
plt.plot(fim_bd, label='backdoor')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('FI', rotation=0)
plt.grid()
plt.savefig(exp_dir+'/fim_badnet.pdf')
plt.show()