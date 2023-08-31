

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy.random as random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wasserstein_distance

font = {'size'   : 28}
matplotlib.rc('font', **font)

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from train_utils.losses import LpLoss

torch.manual_seed(0)
np.random.seed(0)

T = 500
s = 256
S = s

Re = 5000
index = 1
T = 100

HOME_PATH = '/central/groups/tensorlab/miguel/physics_informed/'

############################################################################
# RE500
# dataloader = MatReader(HOME_PATH+'pred/mno5000.mat')

# pretrain = torch.load(HOME_PATH+'pred/re500-1_8s-800-pino-140k-prediction.pt')
# pretrain_truth = pretrain['truth'].squeeze().permute(2,0,1)
# pretrain_pred = pretrain['pred'].squeeze().permute(2,0,1)
#
# finetune = torch.load(HOME_PATH+'pred/re500-1_8s-800-pino-1k-finetune-prediction.pt')
# finetune_truth = finetune['truth'].squeeze().permute(2,0,1)
# finetune_pred = finetune['pred'].squeeze().permute(2,0,1)
#
# fno = torch.load(HOME_PATH+'pred/re500-1_8s-800-fno-50k-prediction.pt')
# fno_truth = fno['truth'].squeeze().permute(2,0,1)
# fno_pred = fno['pred'].squeeze().permute(2,0,1)
#
# fno = torch.load(HOME_PATH+'pred/prediction_unet.pt')
# unet_truth = fno['truth'].squeeze().permute(2,0,1)
# unet_pred = fno['pred'].squeeze().permute(2,0,1)



finetune = torch.load(HOME_PATH + 'exp/Re500-1_8s-800-PINO-s/results/pino-Re500-1_8s-eval.pt')
finetune_truth = finetune['truth'].squeeze().permute(0,3,1,2)
finetune_pred = finetune['pred'].squeeze().permute(0,3,1,2)

fno = torch.load(HOME_PATH + 'exp/Re500-1_8s-800-FNO-s/results/fno-Re500-1_8-eval.pt')
fno_truth = fno['truth'].squeeze().permute(0,3,1,2)
fno_pred = fno['pred'].squeeze().permute(0,3,1,2)

######################## COMMENT OUT BELOW TO USE REGULAR FNO ########################
######## NOTE: PDE Arena FNO only: FFT in space only; must interpolate in time
# fno_pdearena = torch.load(HOME_PATH + 'exp/Re500-1_8s-800-FNO-128-16m/results/fno-prediction.pt')
# fno_pred = fno_pdearena['pred'].squeeze().permute(0,3,1,2)
#####################################################################################

#unet = torch.load(HOME_PATH + 'exp/Re500-1_8s-800-UNet/results/unet-prediction.pt')
#unet = torch.load(HOME_PATH + 'exp/Re500-1_8s-800-U-Netmod-64/results/unet_prediction.pt')
#unet = torch.load(HOME_PATH + 'exp/Re500-1_8s-800-U-Netmod-64/results/pino-loss-prediction.pt')
unet = torch.load(HOME_PATH + 'exp/Re500-1_8s-800-U-FNet1-16m/results/unet-prediction.pt')
#unet = torch.load(HOME_PATH + 'exp/Re500-1_8s-800-UF1Net-16m-PINO/results/temp-prediction2.pt')
#unet = torch.load(HOME_PATH + 'exp/Re500-1_8s-800-U-F1Net-16m-PINO/results/temp-prediction.pt')

#unet = torch.load(HOME_PATH + 'exp/Re500-1_8s-800-U-F1Net-16m-PINO-f_loss-0_3/results/prediction_2000.pt')
#unet = torch.load(HOME_PATH + 'exp/Re500-1_8s-800-U-F1Net-16m-PINO-f_loss-0_01-ic_loss-0_01/results/prediction_2000.pt')
#unet = torch.load(HOME_PATH + 'exp/Re500-1_8s-800-U-F1Net-16m-PINO-f_loss-0_1-ic_loss-0_03/results/prediction_4000.pt')
unet_truth = unet['truth'].squeeze().permute(0,3,1,2)
unet_pred = unet['pred'].squeeze().permute(0,3,1,2)

shape = fno_truth.shape
print(fno_pred.shape, unet_pred.shape)

unet_pred_low = unet_pred.unsqueeze(1)
unet_pred_interp = F.interpolate(unet_pred_low, shape[1:], mode='trilinear').squeeze()

# NOTE: PDE Arena FNO only
fno_pred_low = fno_pred.unsqueeze(1)
fno_pred = F.interpolate(fno_pred_low, shape[1:], mode='trilinear').squeeze()

# Super-resolution loss
lploss = LpLoss(size_average=True)
pino_error = lploss(finetune_pred, fno_truth).item()
pino_error_low_res = lploss(finetune_pred[:, ::2, ::4, ::4], fno_truth[:, ::2, ::4, ::4]).item()
fno_error = lploss(fno_pred, fno_truth).item()
interp_error = lploss(unet_pred_interp, fno_truth).item()

print("PINO super-resolution L2 error:", pino_error)
print("PINO low-resolution L2 error:", pino_error_low_res)
print("FNO super-resolution L2 error:", fno_error)
print("U-Net super-resolution L2 error:", interp_error)
# ##############################################################
### FFT plot
##############################################################

#
def spectrum2(u):
    T = u.shape[0]
    u = u.reshape(T, s, s)
    # u = torch.rfft(u, 2, normalized=False, onesided=False)
    u = torch.fft.fft2(u)
    # ur = u[..., 0]
    # uc = u[..., 1]


    # 2d wavenumbers following Pytorch fft convention
    k_max = s // 2
    wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1), \
                            torch.arange(start=-k_max, end=0, step=1)), 0).repeat(s, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers
    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)
    sum_k = sum_k.numpy()
    # Remove symmetric components from wavenumbers
    index = -1.0 * np.ones((s, s))
    index[0:k_max + 1, 0:k_max + 1] = sum_k[0:k_max + 1, 0:k_max + 1]



    spectrum = np.zeros((T, s))
    for j in range(1, s + 1):
        ind = np.where(index == j)
        # spectrum[:, j - 1] = np.sqrt((ur[:, ind[0], ind[1]].sum(axis=1)) ** 2
                                     # + (uc[:, ind[0], ind[1]].sum(axis=1)) ** 2)
        spectrum[:, j - 1] =  (u[:, ind[0], ind[1]].sum(axis=1)).abs() ** 2


    spectrum = spectrum.mean(axis=0)
    return spectrum



# frame = 64
# pred_sp = spectrum2(pretrain_pred[0:frame+1])
# truth_sp = spectrum2(pretrain_truth[0:frame+1])
# finetune_sp = spectrum2(finetune_pred[0:frame+1])
# fno_sp = spectrum2(fno_pred[0:frame+1])
# unet_interp_sp = spectrum2(unet_pred_interp[0:frame+1])

# pred_sp = spectrum2(pretrain_pred.reshape(50*65, 256,256))
truth_sp = spectrum2(fno_truth.reshape(50*65, 256,256))
finetune_sp = spectrum2(finetune_pred.reshape(50*65, 256,256))
fno_sp = spectrum2(fno_pred.reshape(50*65, 256,256))
print(unet_pred_interp.shape)
unet_interp_sp = spectrum2(unet_pred_interp.reshape(50*65, 256,256))

# np.save('exp/truth_sp.npy', truth_sp)
# np.save('exp/pino_finetune_sp.npy', finetune_sp)
# np.save('exp/fno_sp.npy', fno_sp)
# np.save('exp/unet_interp_sp.npy', unet_interp_sp)

# print(pred_sp.shape)
fig, ax = plt.subplots(figsize=(10,10))

linewidth = 3
ax.set_yscale('log')
# ax.set_xscale('log')

length = 128
k = np.arange(length) * 1.0
#k3 = k**-3 * 100000000000
#k5 = k**-(5/3) * 5000000000
# ax.plot(pred_sp, 'r',  label="pino", linewidth=linewidth)
ax.plot(unet_interp_sp, 'r',  label="U-FNet1-16m + interp.", linewidth=linewidth)
#ax.plot(unet_interp_sp, 'r',  label="U-Netmod-64 + interp.", linewidth=linewidth)
ax.plot(fno_sp, 'b',  label="FNO (Li et al., 2021)", linewidth=linewidth)
#ax.plot(fno_sp, 'b',  label="FNO-128-16m + interpolate", linewidth=linewidth)
ax.plot(finetune_sp, 'g',  label="PINO (Li et al., 2021)", linewidth=linewidth)
ax.plot(truth_sp, 'k', linestyle=":", label="Ground Truth", linewidth=4)
ax.axvline(x=32, color='grey', linestyle='--', linewidth=linewidth)
# ax.plot(k, k5, 'k--',  label="k^-5/3 scaling", linewidth=linewidth)

# ax.set_xlim(1,length)
ax.set_xlim(1,80)
# ax.set_ylim(1,10000000000)
ax.set_ylim(10000,10000000000)
# ax.set_yticks([0.05,0.10,0.15])
#
plt.legend(prop={'size': 20})
# plt.title('averaged over t=[0,'+str(frame)+']' )
plt.title('spectrum of Kolmogorov Flows' )

plt.xlabel('wavenumber')
plt.ylabel('energy')

leg = plt.legend(loc='best')
leg.get_frame().set_alpha(0.5)
# plt.show()
# plt.savefig('re5000-sp-truth-t'+str(frame)+'.png')

plt.tight_layout()
plt.savefig('figures/pdf/KF_spectrum_comparison_FNO_U_F1Net_16m.pdf', format='pdf')
