import proplot as pplt
import matplotlib 
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import scipy.linalg

def M(axis, theta):
  return scipy.linalg.expm(np.cross(np.eye(3), theta*axis/np.linalg.norm(axis)))

class Arrow3D(FancyArrowPatch):
  def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
    super().__init__((0, 0), (0, 0), *args, **kwargs)
    self._xyz = (x, y, z)
    self._dxdydz = (dx, dy, dz)

  def draw(self, renderer):
    x1, y1, z1 = self._xyz
    dx, dy, dz = self._dxdydz
    x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

    xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
    self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
    super().draw(renderer)

  def do_3d_projection(self, renderer=None):
    x1, y1, z1 = self._xyz
    dx, dy, dz = self._dxdydz
    x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

    xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
    self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

    return np.min(zs)

def plotq(ax, eJs, J2invs, res=100, caz=0, calt=0):
  thetacount = res
  phicount = res

  colormap = pplt.Colormap(("light gray", "marine", "powder blue"), name="qplot")
#  colormap = pplt.Colormap(("light gray", "maroon", "light tan"), name="qplot")

  phi = np.linspace(0, 2*np.pi, phicount)
  theta = np.linspace(0, np.pi, thetacount)
  x = np.outer(np.cos(phi), np.sin(theta))
  y = np.outer(np.sin(phi), np.sin(theta))
  z = np.outer(np.ones(phi.shape), np.cos(theta))

  r = np.stack([x, y, z], axis=-1)

  qs = 0
  for i in range(len(eJs)):
    eJ = eJs[i]
    J2inv = J2invs[i]

    xmu = r - eJ.reshape((1, 1, 3))
    qs += np.exp(-0.5*np.einsum("abi,ij,abj->ab", xmu, J2inv, xmu))
  
  facecolors = colormap(qs)

  ax.view_init(elev=calt, azim=caz)
 
  lightsource = matplotlib.colors.LightSource(azdeg=60, altdeg=30)

  ax.plot_surface(x, y, z,
    rcount=thetacount, ccount=phicount,
    edgecolor="none", linewidth=0, antialiased=False,
    facecolors=facecolors, lightsource=lightsource, rasterized=True)
 
  ax.plot(np.sin(phi), np.cos(phi), np.zeros(phi.shape), ls="--", zorder=10, color="black", linewidth=0.5, alpha=0.6)
  ax.plot(np.sin(2*theta), np.zeros(theta.shape), np.cos(2*theta), ls="--", zorder=10, color="black", linewidth=0.5, alpha=0.6)

  ax.set_axis_off()

  ls = (-1, 1)
  ax.set(xlim3d=ls, ylim3d=ls, zlim3d=ls, box_aspect=(1, 1, 1))

def plot_squeezing(
  # matplotlib axes (must be 3D)
  ix,
  # Rotate around mean spin vector
  ts = [0],
  # Position of mean spin vector
  thetas = [np.pi/2],
  phis = [0],
  # Noise squeezing
  xis = [0.3],
  # Base variance
  var0s = [20],
  # Resolution of the sphere. Use <50 for quick rendering, and >500 for high quality
  res = 100,
  # Camera orientation
  caz = 0,
  calt = 0,
  interac_arrows = False,
  interac_f = 1,
  axes_label = False,
  axes_label_pos = [1,0.9,1]
):
  eJs = []
  J2invs = []
  for i in range(len(ts)):
    t = ts[i]
    theta = thetas[i]
    phi = phis[i]
    xi = xis[i]
    var0 = var0s[i]

    eJ = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    Jperp1 = np.cross(eJ, [0, 0.5, 0.5])
    Jperp1 = M(eJ, t) @ Jperp1 / np.linalg.norm(Jperp1)
    Jperp2 = np.cross(eJ, Jperp1)
    J2inv = var0*np.outer(eJ, eJ) + (var0*xi**2)*np.outer(Jperp1, Jperp1) + (var0/xi**2)*np.outer(Jperp2, Jperp2)
    
    eJs.append(eJ)
    J2invs.append(J2inv)

  plotq(ix, eJs, J2invs, res=res, caz=caz, calt=calt)

  if interac_arrows:
    narrows = 10
    zs = np.linspace(-0.5, 0.5, narrows)
    ix.quiver(np.ones(narrows), np.zeros(narrows), zs,
      np.zeros(narrows), interac_f*zs, np.zeros(narrows),
      normalize=False, length=1, linewidth=0.5, color="black", alpha=0.75)

  if axes_label:
    l = 0.4
    dl = 0.05
    p = axes_label_pos
    xarrow = Arrow3D(p[0] - dl, p[1], p[2], l, 0, 0)
    ix.add_artist(xarrow)
    ix.text(p[0] + l + dl, p[1], p[2], r'$x$', ha="right", va="top", fontsize="x-small")
    yarrow = Arrow3D(p[0], p[1] - dl, p[2], 0, l, 0)
    ix.add_artist(yarrow)
    ix.text(p[0], p[1] + l + dl, p[2], r'$y$', ha="center", va="center", fontsize="x-small")
    zarrow = Arrow3D(p[0], p[1], p[2] - dl, 0, 0, l)
    ix.add_artist(zarrow)
    ix.text(p[0], p[1], p[2] + l + dl, r'$z$', ha="center", va="center", fontsize="x-small")

if __name__ == "__main__":
  fig, ax = pplt.subplots(refaspect=1)

  ix = ax.inset([0, 0, 1, 1], proj="3d")

  thetas = np.linspace(0, np.pi, 10)

  plot_squeezing(
    ix,
    ts = np.zeros(thetas.shape),
    thetas = thetas,
    phis = np.cos(thetas),
    xis = np.ones(thetas.shape),
    var0s = 200*np.ones(thetas.shape)
  )
  pplt.show()

