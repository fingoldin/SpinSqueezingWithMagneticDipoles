import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
import numpy as np
from interacs import CG

fig = plt.figure(figsize=(3, 6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0.25, 0.7)
ax.set_ylim(0, 1)
ax.set_aspect(1)
ax.set_axis_off()
ax.grid(visible=False)

nFs = 8
F_spacing = 0.1
F_center0 = np.array([0.5, 0.15])
level_w = 0.2
level_color = "C0"
text_offset = 0.06
sx_theta = 45 # degrees

J0 = 0.934

for Fi in range(nFs):
  F_center = F_center0 + np.array([0, F_spacing*Fi])

  ax.text(F_center[0] - level_w/2 - text_offset, F_center[1], "F=%d/2" % (19 - 2*Fi), fontsize=12, ha="center", va="center")

  ax.plot([F_center[0] - level_w/2, F_center[0] + level_w/2], [F_center[1], F_center[1]], color=level_color)
  
  if Fi < nFs - 1:
    J = J0*np.abs(CG(1/2, (17 - 2*Fi)/2, 6, 7/2))**2
    ax.text(F_center[0], F_center[1] + F_spacing/2, "%.2f Hz" % J, ha="center", va="center", fontsize=16)

    lw = J / 0.6
    ax.add_patch(Arc(F_center + np.array([0, F_spacing/2]), level_w, level_w, theta1=-sx_theta/2, theta2=sx_theta/2, linewidth=lw))
    ax.add_patch(Arc(F_center + np.array([0, F_spacing/2]), level_w, level_w, theta1=180 - sx_theta/2, theta2=180 + sx_theta/2, linewidth=lw))

plt.savefig("figures/interac_level.pdf")
plt.savefig("figures/interac_level.png", dpi=300)
