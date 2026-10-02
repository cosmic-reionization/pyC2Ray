import numpy as np
import matplotlib.pyplot as plt

path = '/scratch/snx3000/mibianco/results_c2ray/test_eurohack23/'

#with open(path+"IonRates_2.000.pkl","rb") as f:
#    data = pkl.load(f)

d1 = np.load(path+'phi_ion_mpi.npy')
d2 = np.load(path+'phi_ion_gpu.npy')

i_plot = 50
fig, axs = plt.subplots(figsize=(20, 10), ncols=3, nrows=1)
axs[0].imshow(d1[50], norm='log')
axs[1].imshow(d2[50], norm='log')
im = axs[2].imshow(d2[50] - d1[50], norm='log')
plt.colorbar(im, ax=axs[2])

plt.savefig(path+'test.png', bbox_inches='tight')
plt.clf()