import proplot as pplt
import numpy as np
from qplots2 import plot_squeezing

fig, ax = pplt.subplots(refaspect=1.5, refwidth=2)
ax.set_axis_off()

res = 500
q_s = 0.7

xi = np.sqrt(0.1)
offset = np.pi/4 - 15*np.pi/180

ix = ax.inset([-q_s/2, 0, q_s, q_s], proj="3d")
plot_squeezing(ix, ts=[0], xis=[1], res=res, calt=10, caz=10, axes_label=False, interac_f=-1)

ix = ax.inset([0, 0, q_s, q_s], proj="3d")
plot_squeezing(ix, ts=[offset], xis=[xi], res=res, calt=10, caz=10, axes_label=True, interac_arrows=True, interac_f=-1)

ix = ax.inset([q_s/2, 0, q_s, q_s], proj="3d")
plot_squeezing(ix, ts=[np.pi/4], xis=[xi], res=res, calt=10, caz=10, axes_label=False, interac_f=-1)

fig.savefig("figures/squeezing_process.pdf")
fig.savefig("figures/squeezing_process.png")
