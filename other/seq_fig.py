import numpy as np
import proplot as pplt
from matplotlib.patches import Rectangle
import matplotlib.colors as colors
from qplots2 import plot_squeezing

c0 = "black"

last_x = 0

def sweep(s, l, f1, f2, c, nump=1000):
  global last_x

  x0 = last_x + s
  x1 = x0 + l

  # ensure the sweeps ends at 0
  x1 -= (f2*(x1 - x0) % (2*np.pi))/f2

  x = np.linspace(x0, x1, nump)
  y = np.sin((f1 + (x - x0)*(f2 - f1)/(x1 - x0))*(x - x0))

  pplt.plot([last_x, x[0]], [0,0], color=c0)
  last_x = x[-1]

  pplt.plot(x, y, color=c)

def box(ax, s, l, c, a=1, nump=1000, label=None, label_rot="horizontal", label_size=8, fc=None):
  global last_x
  
  x0 = last_x + s
  x1 = x0 + l

  ax.plot([x0, x0, x1, x1], a*np.array([0, 1, 1, 0]), color=c)

  if fc is None:
      fc = colors.to_rgb(c)
      fc += (0.5,)

  ax.add_patch(Rectangle([x0, 0], x1 - x0, 1, color=fc))
  last_x = x1

  if label is not None:
    ax.text((last_x + x0)/2, a/2, label, ha="center", va="center", fontsize=0.8*label_size, rotation=label_rot)

def pulse_f(xc, sig, f, c, nump=1000, nsig=6):
  global last_x
  
  xc = last_x + xc

  # ensure the sweeps ends at 0
  x = np.linspace(xc - nsig*sig, xc + nsig*sig, nump)
  y = np.exp(-(x-xc)**2/(2*sig**2))*np.sin(f*x)
  
  pplt.plot([last_x, x[0]], [0,0], color=c0)
  last_x = x[-1]

  pplt.plot(x, y, color=c)

def pulse(xc, sig, c, nump=1000, nsig=3, a=1):
  global last_x
  
  xc = last_x + xc

  # ensure the sweeps ends at 0
  x = np.linspace(xc - nsig*sig, xc + nsig*sig, nump)
  y = a*np.exp(-(x-xc)**2/(2*sig**2))
  
  pplt.plot([last_x, x[0]], [0,0], color=c0)
  last_x = x[-1]

  pplt.plot(x, y, color=c)

fig, axs = pplt.subplots([[1], [2], [3]], refwidth=4, hratios=(1, 0.4, 0.4), refheight=0.5, hspace=(1, 0.5))

axs[0].plot([0], [0])

mF_f = 20
main_f = 100

s = 0.5
last_x = s

# Sweep to mF=1/2
box(axs[0], 0, 1.6, "black", fc="white", label=r'$\mathrm{F}=\frac{19}{2},$' "\n" r'$\mathrm{m}_{\mathrm{F}}=-\frac{19}{2}$' "\nprep", label_size=6)
box(axs[0], 0.5, 1.4, "orange9", label=r'$\mathrm{m}_{\mathrm{F}}$' "\n" r'$-\frac{19}{2}\to\frac{1}{2}$', label_size=7)
# Sweep to F=17/2
box(axs[0], 0.5, 1.2, "C2", label=r'$\mathrm{F}$' "\n" r'$\frac{19}{2}\to\frac{17}{2}$', label_size=6)
# Blowout
box(axs[0], 0.2, 0.4, "yellow7")
# pi/2
box(axs[0], 0.2, 0.6, "blue7", label=r'$\frac{\pi}{2}_y$')
# pi
box(axs[0], 0.4, 0.8, "blue7", label=r'$\pi_y$')
# pi
box(axs[0], 0.9, 0.8, "blue7", label=r'$\pi_{-y}$')
# readout pulse
box(axs[0], 0.4, 0.6, "blue7", label=r'$\phi_x$')
# Sweep to mF=-17/2
box(axs[0], 0.75, 1.3, "C1", label=r'$\mathrm{m}_\mathrm{F}=\frac{-17}{2}$' "\n" "shelve", label_size=6)
box(axs[0], 0.2, 1.3, "C1", label=r'$\mathrm{m}_\mathrm{F}=\frac{19}{2}$' "\n" "shelve", label_size=6)
# Image
box(axs[0], 0.8, 1, "lime7", label="Image\n" r'$F=\frac{19}{2}$', label_size=6)
# Sweep to F=19/2
box(axs[0], 0.2, 1.2, "C2", label=r'$\mathrm{F}$' "\n" r'$\frac{17}{2}\to\frac{19}{2}$', label_size=6)
# Image
box(axs[0], 0.3, 1, "lime7", label="Image\n" r'$F=\frac{19}{2}$', label_size=6)
box(axs[0], 0.2, 1.2, "C2", label=r'$\mathrm{F}$' "\n" r'$\frac{17}{2}\to\frac{19}{2}$', label_size=6)
box(axs[0], 0.3, 1, "lime7", label="Image\n" r'$F=\frac{19}{2}$', label_size=6)

xmax = last_x + 0.5

axs[0].plot([0, xmax], [0, 0], color=c0, alpha=0.5)
#plt.ylabel("Pulse sequence", labelpad=32)

#plt.subplot(212)

Bcolor = "grey8"
Zcolor = "red7"

axs[2].plot([
  0, 2.1, 2.2, 2.35, # mF sweep
], [
  0, 0, 0, 18,
], color=Bcolor)
axs[2].plot([  
  4.35, 4.45,
], [
  19, 10.6
], color=Bcolor)
axs[2].plot(np.array([
  4.45, 10.92 + 0.6,
]), [10.6, 10.6], color=Zcolor)
axs[2].plot(0.6 + np.array([
   10.92, 10.98
]), [10.6, 20], color=Bcolor)
axs[2].plot(np.array([
  14.28 + 0.6, 14.48 + 0.6, xmax
]), [
  20, 1, 1
], color=Bcolor)

axs[1].plot([
  2.38, 2.48, 4.12, 4.32
], [
  55, 59, 61, 55
], color=Bcolor)
  

axs[1].plot(9.2 + np.array([
  2.38, 2.48, 3.98, 5.48, 5.68
]), [
  55, 61, 59, 61, 55
], color=Bcolor)
#21, 10.6,
#  10.6, 0, 0#, 56, 54, 0
  
  #4.1, 4.4, # mag insens
 # 10.4, 10.5, xmax #, 10.8, 11.9, 12.2

axs[2].text(5.5, 13, "Magnetically insensitive field", fontsize=5, color=Zcolor)

axs[0].format(yticks=[])
axs[1].format(yticks=[0, 59, 61], ylim=(55, 62), yminorlocator=[], ylabel="B field (G)", yticklabelsize=5, ylabelsize=5)
axs[2].format(yticks=[0, 10.6], ylim=(-2, 19), yminorlocator=[], yticklabelsize=5)
#axs[3].format(yticks=[])
axs[1].spines["bottom"].set_visible(False)
axs[2].spines["top"].set_visible(False)
axs[2].spines["top"].set_visible(False)

fig.format(
  xlocator=[0.3, 2.2, 4.4, 5.8, 10.4, 11.6, 13.1, xmax-0.3], xminorlocator=[],
  xformatter=["0", "7.5", "7.8", "7.85", "8", "8.2", "8.25", "8.3"], xlabel="Time (s), not to scale",
  xlabelsize=5,
  xticklabelsize=5
)


res = 500
q_s = 1.3

xi = np.sqrt(0.1)
offset = np.pi/4 - 15*np.pi/180

ss = -0.27

#ix = axs[3].inset([ss - 0.13, -0.22, q_s, q_s], proj="3d")
#plot_squeezing(ix, ts=[0], xis=[1], res=res, calt=10, caz=10, axes_label=True, interac_f=-1)

#ix = axs[3].inset([ss, -0.22, q_s, q_s], proj="3d")
#plot_squeezing(ix, ts=[offset], xis=[xi], res=res, calt=10, caz=10, axes_label=True, interac_arrows=True, interac_f=-1)

#ix = axs[3].inset([ss + 0.13, -0.22, q_s, q_s], proj="3d")
#plot_squeezing(ix, ts=[np.pi/4], xis=[xi], res=res, calt=10, caz=10, axes_label=True, interac_f=-1)


"""
plt.legend()
plt.xlim(0, xmax)
plt.ylabel("Magnetic field (G)")
plt.yticks([0, 10, 60])
plt.xticks([])
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)

plt.tight_layout()
"""
fig.savefig("figures/seq_fig.png", dpi=300)
fig.savefig("figures/seq_fig.pdf")
