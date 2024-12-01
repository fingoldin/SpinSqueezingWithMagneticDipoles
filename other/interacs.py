import math
import numpy as np
import proplot as pplt

muB = 9.2740100783e-24
h = 6.62607015e-34
hbar = h / (2*np.pi)
mu0 = 1.25663706212e-6
r_a = 266e-9

def get_gJ(J, S, L):
  return 1 + (J*(J+1) + S*(S+1) - L*(L+1))/(2*J*(J+1))

def choose(n, k):
  n = int(n)
  k = int(k)
  if n < k or k < 0:
    return 0
  if n == k or k == 0:
    return 1
  return math.comb(n, k)

def sign(n):
  if n > 0:
    return 1
  if n < 0:
    return -1
  return 0

def Clebsch(j1, j2, m1, m2, j, m):
  if m1 + m2 != m or abs(m1) > j1 or abs(m2) > j2 or abs(m) > j or j > j1 + j2 or j < abs(j1 - j2):
    return 0
  denom = choose(j1+j2+j+1,j1+j2-j)*choose(2*j1,j1-m1)*choose(2*j2,j2-m2)*choose(2*j,j-m)
  sumterm = sum([ (-1)**a*choose(j1+j2-j,a)*choose(j1-j2+j,j1-m1-a)*choose(-j1+j2+j,j2+m2-a) for a in range(int(j1-m1+1)) ])
  return sign(sumterm)*math.sqrt(choose(2*j1, j1+j2-j)*choose(2*j2, j1+j2-j)*sumterm**2/denom)

def CG(mF, F, J, I):
  return sum([ mJ*Clebsch(J, I, mJ, mF-mJ, F, mF)*Clebsch(J, I, mJ, mF-mJ, F+1, mF) for mJ in np.arange(-J, J+1, 1) ])

def maxCG(J, I):
  return max([
    max([ CG(mF, F, J, I) for mF in np.arange(-F, F+1, 1) ]) for F in np.arange(np.abs(J-I), J+I, 1)
  ])

if __name__ == "__main__":
  fig, ax = pplt.subplots(refaspect=2, refwidth=2)

  gJs = np.array([2, 1.16, 1.24, 
    #1.25,
    get_gJ(2, 1, 1), 1.16, 1.463])

  IJs = np.array([
    [3/2, 1/2], # Rb87
    [7/2, 6], # Er167
    [5/2, 8], # Dy
    #[7, 8], # Ho166m1
    [9/2,2], # Sr87 
    [7/2, 6], # Er167
    [5/2, 2] # Yb173
  ])

  """ 
  shifts = np.array([
    [-0.1, 0],
    [0.5, -0.5],
    [-0.2, 0.7],
    [1.5, 1],
    [1.2, -0.5],
    [1.2, 0]
  ])
  """
  shifts = np.array([
    [-0.05, 0],
    [0.8, 0.3],
    [-0.05, 0],
   # [0, 0],
    [-0.05, 0],
    [1.15, 0],
    [0.65, 0.3]
  ])

  colors = ["C0", "C0", "C0", "C0", "C1", "C0"]

  J0 = mu0*muB**2/(4*np.pi*r_a**3*2*np.pi*hbar)
  Js = J0*gJs**2*np.array([ maxCG(IJ[1], IJ[0]) if (IJ[0] != 0 and IJ[1] != 0) else 0 for IJ in IJs ])**2/2

  # This work
  Js[-2] = J0*gJs[-2]**2*CG(1/2, 17/2, 6, 7/2)**2/2

  for i in range(len(Js)):
    ax.scatter(np.min(IJs, axis=1)[i], 2*Js[i], m=".", color=colors[i])
  
  print(Js)

#  names = [r'$^{87}$Rb', r'$^{167}$Er', r'$^{163,161}$Dy', r'$^{166\mathrm{m1}}$Ho', r'$^{87}Sr ^{3}P_2$']
  names = [r'Rb$^{87}$', r'max Er$^{167}$', r'Dy$^{161,163}$',
    #r'Ho',
    r'Sr$^{87}$', r'This work', r'Yb$^{173}$']
  
  for i, txt in enumerate(names):
    pos = np.array([np.min(IJs, axis=1)[i], 2*Js[i]])
    ax.annotate(txt, pos, pos - shifts[i]*np.array([1,2]))

  ax.format(xlabel=r'$\mathrm{min}\{I, J\}$', ylabel=r'$J_{\perp}$ (Hz)', ylim=(0, 4), xlim=(0, 4), tickdir="in", xminorticks=[], yminorticks=[])
  fig.savefig("figures/CGs.png", dpi=600)
  fig.savefig("figures/CGs.pdf")
