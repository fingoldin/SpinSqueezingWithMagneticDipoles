import numpy as np
import sys
try:
  import tensorflow as tf
except:
  print("Failed to load tensorflow")
  cp = None
import warnings

def get_rpoints(s0, corr=False, l=None):
  s0 = s0 / np.linalg.norm(s0)

  x = np.random.randn(3)
  x = x / np.linalg.norm(x)

  fc = l if l is not None else (1 if corr else np.sqrt(2))
  print("fc:", fc, flush=True)

  ortho1 = np.cross(s0, x)
  ortho1 = fc*ortho1/np.linalg.norm(ortho1)
  ortho2 = np.cross(s0, ortho1)
  ortho2 = fc*ortho2/np.linalg.norm(ortho2)

  return np.stack([
    s0 + ortho1,
    s0 - ortho1,
    s0 + ortho2,
    s0 - ortho2
  ], axis=0)

s2mask = np.array([
  np.eye(3)/4
])

def crossidx(M, usetf=False):
  xp = tf if usetf else np
  return xp.stack([
    M[...,1,2] - M[...,2,1],
    M[...,2,0] - M[...,0,2],
    M[...,0,1] - M[...,1,0],
  ], axis=-1)

def cross(a, b, axis=-1, usetf=False):
  xp = tf if usetf else np
  if not usetf:
    return np.cross(a, b, axis=axis)

  if len(a.shape) != len(b.shape):
    raise ValueError("cross inputs must have same shape len")

  if axis < 0:
    axis = len(a.shape) + axis
  
  tr = list(range(len(a.shape)))
  tr[-1] = axis
  tr[axis] = len(a.shape) - 1

  a = xp.transpose(a, tr)
  b = xp.transpose(b, tr)

  return xp.transpose(xp.stack([
    a[...,1]*b[...,2] - a[...,2]*b[...,1],
    a[...,2]*b[...,0] - a[...,0]*b[...,2],
    a[...,0]*b[...,1] - a[...,1]*b[...,0]
  ], axis=-1), tr)


# Get the averaged moments of the spin values
def get_moments(s, c, dtype, usetf=False, retallts=False, retallsites=True, retallsites2=True, corr=False, mask=None, diff_mask=None):
  nt = s.shape[0]
  xp = tf if usetf else np
  rsum = xp.reduce_sum if usetf else xp.sum

  do_diff = mask is not None and diff_mask is not None

  if not do_diff and mask is not None:
    if not usetf:
      s = s[:,mask,:]
      if len(c.shape) == 5:
        c = c[:,mask,:,:,:][:,:,mask,:,:]
    else:
      s = tf.boolean_mask(s, mask, axis=1)
      if len(c.shape) == 5:
        c = tf.boolean_mask(tf.boolean_mask(c, mask, axis=1), mask, axis=2)

  if retallts:
    s1 = s
    s2 = xp.einsum("aij,akl->aikjl", s, s) + c
  else:
    s1 = rsum(s, axis=0)/s.shape[0]
    s2 = (xp.einsum("aij,akl->ikjl", s, s) + rsum(c, axis=0))/nt

  # Fix the diagonals of the operators
  # TODO: test the tensorflow version of this code
  if corr:
    if retallts:
      if usetf:
        s2 -= xp.transpose(xp.linalg.diag(
            xp.linalg.diag_part(xp.transpose(s2, (0, 3, 4, 1, 2)))
          ), (0, 3, 4, 1, 2)) - \
          xp.tile(
            xp.expand_dims(
              xp.transpose(
                xp.reshape(
                  xp.eye(s2.shape[1]*s2.shape[3], s2.shape[2]*s2.shape[4], dtype=dtype)/4,
                  (s2.shape[1], s2.shape[3], s2.shape[2], s2.shape[4])
                ), (0, 2, 1, 3)
              ), axis=0
            ), (s2.shape[0], 1, 1, 1, 1)
          )
      else:
        np.einsum("aiijl->aijl", s2)[:] = np.expand_dims(s2mask, axis=0)
    else:
      if usetf:
        s2 -= xp.transpose(xp.linalg.diag(
            xp.linalg.diag_part(xp.transpose(s2, (2, 3, 0, 1)))
          ), (2, 3, 0, 1)) - \
          xp.transpose(
            xp.reshape(
              xp.eye(s2.shape[0]*s2.shape[2], s2.shape[1]*s2.shape[3], dtype=dtype)/4,
              (s2.shape[0], s2.shape[2], s2.shape[1], s2.shape[3])
            ), (0, 2, 1, 3)
          )
      else:
        np.einsum("iijl->ijl", s2)[:] = s2mask
  
  if not retallsites:
    a = 1 if retallts else 0
    if do_diff:
      mask2 = mask | diff_mask
      if not usetf:
        if retallts:
          s1 = rsum(s1[:,mask2,:], axis=a)
        else:
          s1 = rsum(s1[mask2,:], axis=a)
      else:
        s1 = rsum(tf.boolean_mask(s1, mask2, axis=a), axis=a)
    else:
      s1 = rsum(s1, axis=a)

  if not retallsites2:
    a = (1, 2) if retallts else (0, 1)
    if do_diff:
      if not usetf:
        if retallts:
          s2 = rsum(s2[:,mask,:,:,:][:,:,mask,:,:], axis=a) +\
               rsum(s2[:,diff_mask,:,:,:][:,:,diff_mask,:,:], axis=a) -\
               rsum(s2[:,mask,:,:,:][:,:,diff_mask,:,:], axis=a) -\
               rsum(s2[:,diff_mask,:,:,:][:,:,mask,:,:], axis=a)
        else:
          s2 = rsum(s2[mask,:,:,:][:,mask,:,:], axis=a) +\
               rsum(s2[diff_mask,:,:,:][:,diff_mask,:,:], axis=a) -\
               rsum(s2[mask,:,:,:][:,diff_mask,:,:], axis=a) -\
               rsum(s2[diff_mask,:,:,:][:,mask,:,:], axis=a)
      else:
          s2 = rsum(tf.boolean_mask(tf.boolean_mask(s2, mask, axis=a[0]), mask, axis=a[1]), axis=a)\
            + rsum(tf.boolean_mask(tf.boolean_mask(s2, diff_mask, axis=a[0]), diff_mask, axis=a[1]), axis=a)\
            - rsum(tf.boolean_mask(tf.boolean_mask(s2, mask, axis=a[0]), diff_mask, axis=a[1]), axis=a)\
            - rsum(tf.boolean_mask(tf.boolean_mask(s2, diff_mask, axis=a[0]), mask, axis=a[1]), axis=a)
    else:
      s2 = rsum(s2, axis=a)

  if usetf:
    s1 = s1.numpy()
    s2 = s2.numpy()

  return s1, s2

# Symmetrize M_[nklab] tensors
def symm(M):
  return M + np.transpose(M, (0, 2, 1, 4, 3))

def get_derivs(J, R, s, c, dtype, corr=False, usetf=False):
  xp = tf if usetf else np
  array = (lambda v: xp.constant(v, dtype=dtype)) if usetf else (lambda v: xp.array(v, dtype=dtype))

  N = J.shape[0]
  nt = s.shape[0]

  Rmat = xp.reshape(R, (1, N, 1, 3, 1))
  Jd = xp.reshape(J, (1, N, N, 3, 3))

  xp = tf if usetf else np

  # Effective field on the classical spins
  Jms = xp.einsum("ijkl,njl->nik", J, s) 
  Beff = 2*(R + 4*Jms)
  # Classical spin precession
  sdot = cross(Beff, s, usetf=usetf)

  cdot = array([0])
  if corr:
    # Correction field
    Bceff = 8*xp.einsum("ikab,nikac->nkbc", J, c)
    sdot += crossidx(Bceff, usetf=usetf)

    Jcs = cross(Jd, xp.reshape(s, (nt, N, 1, 3, 1)), axis=-2, usetf=usetf)
    corr1 = -8*xp.einsum("nkla,nlb->nklab", cross(
      xp.einsum("klac,nlc->nkla", J, s), xp.reshape(s, (nt, N, 1, 3)), usetf=usetf
    ), s) + 2*Jcs

    corr2 = 2*cross(Rmat, c, axis=-2, usetf=usetf)
    
    Jms = xp.reshape(Jms, (nt, N, 1, 3, 1))
    Jmsl = xp.reshape(xp.einsum("klac,nlc->nkla", J, s), (nt, N, N, 3, 1))
    Jc = xp.einsum("klac,nklbc->nklab", J, c)
    corr3 = 8*(
      xp.einsum("nkjac,njlcb->nklab", Jcs, c) + cross(Jms, c, axis=-2, usetf=usetf) - (\
        xp.einsum("nklac,nllcb->nklab", Jcs, c) +\
        cross(Jmsl, c, axis=-2, usetf=usetf) +\
        xp.reshape(crossidx(Jc, usetf=usetf), (nt, N, N, 3, 1))*xp.reshape(s, (nt, 1, N, 1, 3))
    )) # expensive

    cdot = corr1 + corr2 + corr3
    cdot = cdot + xp.transpose(cdot, (0, 2, 1, 4, 3))

  del Jms, Beff
  if corr:
    del Bceff, Jcs, corr1, corr2, Jmsl, Jc, corr3

  return sdot, cdot

class Pair:
  def __init__(self, a, b):
    self.a = a
    self.b = b

  def __add__(self, val):
    if isinstance(val, Pair):
      return Pair(self.a + val.a, self.b + val.b)
    return Pair(self.a + val, self.b + val)
  
  def __radd__(self, val):
    return self.__add__(val)
  
  def __mul__(self, val):
    return Pair(val*self.a, val*self.b)
  
  def __rmul__(self, val):
    return self.__mul__(val)
  
  def __truediv__(self, val):
    return self.__mul__(1/val)

def rk4_update(f, x, h):
  a = f(x)
  b = f(x + a*(h/2.0))
  c = f(x + b*(h/2.0))
  d = f(x + c*h)

  return x + h*(a + 2*b + 2*c + d)/6

def parseJ(J, array):
  Jupper = np.transpose(np.triu(np.transpose(J, (2, 3, 0, 1)), 1), (2, 3, 0, 1))
  J = array((Jupper + np.transpose(Jupper, (1, 0, 3, 2)))/2)
  return J

# s0 is the vector of initial spin 1/2 operator values for every spin (all spins are assumed
# to begin polarized in the same direction).
# 
# Only the upper triangular part of J is used: on-diagonal elements are zeroed and the matrix is
# symmetrized (on-diagonal elements lead to complex spin components)
def dtwa(s0, J, R, precision=32, nt=500, tsteps=200, t=1, corr=True, usetf=False, stepfunc=None, verbose=False, userot=False, retallts=False, retallsites=True, retallsites2=True, userk4=True, rl=None, analysis_mask=None, diff_mask=None):
  N = J.shape[0]

  xp = tf if usetf else np
  dtype = getattr(xp, "float" + str(precision))
  print("dtype:", dtype)
  array = (lambda v: xp.constant(v, dtype=dtype)) if usetf else (lambda v: xp.array(v, dtype=dtype))
  barray = (lambda v: xp.constant(v, dtype=bool)) if usetf else (lambda v: xp.array(v, dtype=bool))

  analysis_mask = barray(analysis_mask) if analysis_mask is not None else None

  J = parseJ(J, array)
  
  rpoints = get_rpoints(s0, corr=corr, l=rl)

  ps = (1 + 2*np.dot(rpoints, s0))/4
  ps = ps/np.sum(ps)
  samples = np.random.choice(4, size=(nt, N), p=ps)

  # Construct initial spin states from distribution samples
  s = array(rpoints[samples]/2)

  dt = t / tsteps

  # Correlation corrections
  c = xp.zeros((nt, N, N, 3, 3), dtype=dtype) if corr else array([0])

  R = array(R)
  R = xp.reshape(R, (1, N, 3))
  
  for ti in range(tsteps):
    if verbose:
      print("t=%.4f, step %d/%d" % ((ti + 1)*dt, ti + 1, tsteps), flush=True)

    if userk4:
      u = rk4_update(lambda v: Pair(*get_derivs(J, R, v.a, v.b, dtype, corr=corr, usetf=usetf)), Pair(s, c), dt)
      s = u.a
      c = u.b
    else:
      sdot, cdot = get_derivs(J, R, s, c, dtype, corr=corr, usetf=usetf)
      s += sdot*dt
      c += cdot*dt

    if stepfunc is not None:
      moments = get_moments(s, c, dtype, usetf=usetf, retallts=retallts, retallsites=retallsites, retallsites2=retallsites2, corr=corr, mask=analysis_mask, diff_mask=diff_mask)
      d = stepfunc(ti, dt, *moments, J, usetf=usetf)
      if d is not None:
        if "rot" in d:
          s = s @ d["rot"].T
          if corr:
            c = d["rot"] @ c @ d["rot"].T
        if "J" in d:
          J = parseJ(d["J"], array)

  return get_moments(s, c, dtype, usetf=usetf, corr=corr, retallts=retallts, retallsites=retallsites, retallsites2=retallsites2, mask=analysis_mask, diff_mask=diff_mask)
