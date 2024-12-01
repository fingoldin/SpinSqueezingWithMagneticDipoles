import matplotlib.pyplot
import proplot as pplt
import numpy as np
from qplots2 import plot_squeezing

def chi_oat(aoccs):
    toccs = np.sum(aoccs, axis=0)
    mean_occs = np.mean(toccs, axis=-1)
    mask = mean_occs.flatten() > 0
    occs = toccs.reshape((-1, toccs.shape[-1]))[mask,:]
    
    idxs = np.indices(toccs.shape[0:2]).reshape((2, -1))[:,mask]
    
    diffs = 1/np.linalg.norm(np.expand_dims(idxs, axis=-1) - np.expand_dims(idxs, axis=-2), axis=0)**3
    samesite = np.isnan(diffs) | np.isinf(diffs)
    diffs[samesite] = 0
    
    boccs = occs.astype(bool)
    doccs = np.expand_dims(boccs, axis=0) & np.expand_dims(boccs, axis=1)
    
    adiffs = np.expand_dims(diffs, axis=-1)*doccs
    chi = np.sum(adiffs, axis=(0,1))/np.sum(adiffs != 0, axis=(0,1))
    
    return (2*np.pi)*4*0.543*chi

def oat_squeezing(chi, t, N):
    theta = 2*chi*t
    C = 0.25*(
        np.sqrt(np.power(1 - np.power(np.cos(theta), N-2), 2) + 16*np.power(np.sin(theta/2), 2)*np.power(np.cos(theta/2), 2*N-4)) \
        - (1 - np.power(np.cos(theta), N-2))
    )
    xi2 = 1 - (N-1)*C

    return xi2

fig, ax = pplt.subplots(refaspect=1, refwidth=1.5)
ax.axis("off")

q_res = 500
all_thetas = np.array([60, 90, 120])*np.pi/180
tau = 0
occs = 0.5*np.ones((2, 6, 6, 1))

chi = chi_oat(occs)
N = np.sum(occs)/occs.shape[-1]
xi = np.sqrt(oat_squeezing(chi, tau, N))

ixs = []
for i in range(len(all_thetas)):
    ixs.append(ax.inset([i*0.5, 0, 0.6, 0.6], proj="3d"))

for i, theta in enumerate(all_thetas):
    thetas = np.array([theta])

    main = (i == len(all_thetas)//2)
    plot_squeezing(
      ixs[i],
      ts = 0.5*np.ones(thetas.shape),
      thetas = thetas,
      phis = -61*tau*np.cos(thetas),
      xis = xi*np.ones(thetas.shape),
      var0s = N*np.ones(thetas.shape),
      res = q_res,
      caz = 10, calt = 10, interac_arrows=main,
      interac_f=-1, axes_label=main
    )

fig.savefig("figures/mean_fig2_bloch.pdf")
fig.savefig("figures/mean_fig2_bloch.png", dpi=600)

fig.show()
