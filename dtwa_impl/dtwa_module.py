import numpy as np
import sys
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
import os
import scipy
import pickle
import datetime
from IPython.display import clear_output
import glob
import IPython.display as display
from ipywidgets import Output
from tqdm.notebook import tqdm

import .dtwa_lib as dtwa_lib

def Rz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def Rx(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def Pulse(phase, length):
    return Rz(2 * np.pi * phase) @ Rx(2 * np.pi * length) @ Rz(-2 * np.pi * phase)

class DTWAModule:
    # occupations shape: (nrows, ncols, nshots)
    def __init__(
            self,
            occupations,
            endt,
            nshots=-1,
            nt = 100,
            tsteps=1000,
            pulses=[],
            disorder=None,
            analysis_mask=None,
            diff_mask=None,
            retallsites=False,
            verbose=False,
            saveprefix=None,
            savedir=None,
            tunnel_freq=0,
            tunnel_type="simple",
            tunnel_filter_sigma = 1,
            avg_all_Js = False,
            s0 = np.array([1, 0, 0])/2
           ):

        if tunnel_freq > 0 and disorder is not None:
            raise ValueError("Simultaneous disorder and tunneling are currently not supported")
        
        if avg_all_Js and (tunnel_freq > 0 or disorder is not None):
            raise ValueError("Simultaneous avg_all_Js and tunneling/disorder are currently not supported")
            
        if analysis_mask is not None and diff_mask is not None and \
            np.sum(diff_mask.flatten().astype(bool) & analysis_mask.flatten().astype(bool)) > 0:
            raise ValueError("analysis_mask and diff_mask cannot have any overlap")

        if len(occupations.shape) == 2:
          occupations = np.expand_dims(occupations, axis=-1)

        self.occupations = occupations
        self.endt = endt
        self.nshots = nshots
        self.nt = nt
        self.tsteps = tsteps
        self.pulses = pulses
        self.retallsites = retallsites
        self.disorder = disorder
        self.analysis_mask = analysis_mask
        self.diff_mask = diff_mask
        self.verbose = verbose
        self.savedir = savedir
        self.saveprefix = saveprefix
        self.s0 = s0
        self.tunnel_freq = tunnel_freq
        self.tunnel_type = tunnel_type
        self.tunnel_filter_sigma = tunnel_filter_sigma
        self.tunnel_P = scipy.ndimage.gaussian_filter(np.mean(self.occupations, axis=-1), self.tunnel_filter_sigma).flatten()
        self.avg_all_Js = avg_all_Js

    def simulate(self, debugfunc=None):
        out_S, out_S2, out_s1, ts, atom_masks = self._run_shots(self.nshots, self.occupations, self.endt, self.nt, self.disorder, self.retallsites, self.pulses, self.tsteps, self.verbose, self.savedir, self.saveprefix, self.analysis_mask, self.diff_mask, self.s0, debugfunc)

        self.out_S = out_S
        self.out_s1 = out_s1
        self.out_S2 = out_S2
        self.out_ts = np.array(ts)
        self.out_atom_masks = atom_masks

    def analyze(self, nphipoints=180, phiend=180, retdata=False, retdataS=False, plot=True):
        if plot:
          fig, axs = plt.subplots(4, 1, sharex=False, sharey=False)

        phis = tf.linspace(0.0, phiend*np.pi/180, nphipoints)
        Ts, Phis = np.meshgrid(self.out_ts, phis, indexing="ij")

        S = self.out_S
        S2 = self.out_S2

        print(r'Calculating xi^2')
        DeltaJhat2s = []
        for shot_idx in tqdm(range(S.shape[0])):
            DeltaJhat2s.append([])
            for i in range(Ts.shape[0]):
                DeltaJhat2s[-1].append(self._get_DeltaJhat2s(S[shot_idx,i], S2[shot_idx,i], phis))
        DeltaJhat2s = np.array(DeltaJhat2s)

        if self.analysis_mask is not None:
          Ns = []
          if self.diff_mask is not None:
            for shot_idx in range(self.out_atom_masks.shape[0]):
              Ns.append([
                np.sum(np.array(self.analysis_mask).astype(bool).flatten()[self.out_atom_masks[shot_idx,:]]),
                np.sum(np.array(self.diff_mask).astype(bool).flatten()[self.out_atom_masks[shot_idx,:]])
              ])
          else:
            for shot_idx in range(self.out_atom_masks.shape[0]):
              Ns.append(np.sum(np.array(self.analysis_mask).astype(bool).flatten()[self.out_atom_masks[shot_idx,:]]))
          Ns = np.array(Ns)
          print("Ns in analyze:", Ns)
        else:
          Ns = np.sum(self.out_atom_masks, axis=1)
        
        totalNs = np.sum(Ns, axis=1) if self.diff_mask is not None else Ns
        
        eJs = np.linalg.norm(S, axis=-1)/np.expand_dims(totalNs/2, axis=1)
        
        xi2s = DeltaJhat2s / (np.expand_dims(totalNs, axis=(1,2))/4)

        if plot:
          c = axs[0].pcolormesh(Ts, Phis*180/np.pi, 10*np.log10(np.mean(xi2s, axis=0)))
          fig.colorbar(c, cax=axs[0])
          axs[0].set_ylabel("Readout phi (deg)")
          axs[0].set_xlabel("Evolution time (s)")
          axs[0].set_title(r'$\langle\xi^2\rangle$ (dB)')

        mueJs = np.mean(eJs, axis=0)
        stdeJs = np.std(eJs, axis=0, ddof=1) if eJs.shape[0] > 1 else 0
        if plot:
          axs[1].fill_between(self.out_ts, mueJs - stdeJs, mueJs + stdeJs)
          axs[1].set_xlabel("Evolution time (s)")
          axs[1].set_ylabel("Contrast")

        eMs = np.sqrt(S2[:,:,0,0]+S2[:,:,1,1])/np.expand_dims(totalNs/2, axis=1)
        mueMs = np.mean(eMs, axis=0)
        stdeMs = np.std(eMs, axis=0, ddof=1) if eJs.shape[0] > 1 else 0
        if plot:
          axs[2].fill_between(self.out_ts, mueMs - stdeMs, mueMs + stdeMs)
          axs[2].set_xlabel("Evolution time (s)")
          axs[2].set_ylabel("Magnetization")
        
        if self.analysis_mask is not None:
          an_mask = np.expand_dims(np.array(self.analysis_mask).flatten(), axis=0).astype(bool)
          if self.diff_mask is not None:
            an_mask_diff = np.expand_dims(np.array(self.diff_mask).flatten(), axis=0).astype(bool)
            if plot:
              c = axs[3].imshow(
                np.mean(an_mask & self.out_atom_masks, axis=0).reshape(self.occupations.shape[:2]).T -\
                np.mean(an_mask_diff & self.out_atom_masks, axis=0).reshape(self.occupations.shape[:2]).T  ,
                origin="lower"
              )
          else:
            if plot:
              c = axs[3].imshow(
                np.mean(an_mask & self.out_atom_masks, axis=0).reshape(self.occupations.shape[:2]).T,
                origin="lower"
              )
        else:
          if plot:
            c = axs[3].imshow(np.mean(self.out_atom_masks, axis=0).reshape(self.occupations.shape[:2]).T, origin="lower")
        if plot:
          fig.colorbar(c, cax=axs[3])
          axs[3].set_xlabel("x sites")
          axs[3].set_ylabel("y sites")

        if retdata:
            return self.out_ts, phis, eJs, xi2s, Ns
        if retdataS:
            return self.out_ts, phis, self.out_s1, self.out_S2

    def _get_DeltaJhat2s(self, eJvec, eJ2, phi):
        array = (lambda v: tf.constant(v, dtype=tf.float32))

        eJvec = array(eJvec)
        eJ2 = array(eJ2)

        eJhat = eJvec / tf.linalg.norm(eJvec)

        Jhat1 = tf.linalg.cross(eJhat, [0, 0, 1])
        if tf.linalg.norm(Jhat1) == 0:
            Jhat1 = tf.constant([1, 0, 0])
        else:
            Jhat1 = Jhat1 / tf.linalg.norm(Jhat1)
        Jhat2 = tf.linalg.cross(eJhat, Jhat1)

        Jhats = tf.einsum("a,i->ai", tf.sin(phi), Jhat1) + tf.einsum("a,i->ai", tf.cos(phi), Jhat2)

        DeltaJhat2s = tf.maximum(0, tf.reduce_sum(tf.einsum("ai,aj->aij", Jhats, Jhats) * tf.expand_dims(eJ2, axis=0),
                                    axis=(1, 2)) \
                                    - tf.einsum("ai,i->a", Jhats, eJvec) ** 2
                                )

        return DeltaJhat2s.numpy()

    def _run_shots(self, nshots, occupations, endt, nt, disorder, retallsites, pulses, tsteps, verbose, savedir, saveprefix, analysis_mask, diff_mask, s0, debugfunc):
        out_S = []
        out_S2 = []
        out_s1 = []
        atom_masks = []
        idx = 0

        out = Output()
        display.display(out)

        for i in tqdm(range(nshots)):
            with out:
                saveprefixshot = saveprefix + "___shot" + str(idx) if (saveprefix is not None and savedir is not None) else None

                f = glob.glob(os.path.join(savedir, saveprefixshot + "*")) if saveprefixshot is not None else []
                if len(f) > 0:
                    print("Loading from file")
                    data = pickle.load(open(f[0], "rb"))
                    all_S = data["data"]["all_S"]
                    all_S2 = data["data"]["all_S2"]
                    all_s1 = data["data"]["all_s1"]
                    atom_mask = data["data"]["atom_mask"]
                    ts = data["data"]["ts"]
                else:
                    i = np.random.randint(occupations.shape[2])
                    print("Shot index %d" % i)

                    all_S, all_S2, all_s1, ts, atom_mask = self._run_dtwa(occupations[:,:,i], endt, nt, disorder, retallsites, pulses, tsteps, verbose, savedir, saveprefixshot, analysis_mask, diff_mask, s0, debugfunc)

                out_S.append(all_S)
                out_S2.append(all_S2)
                out_s1.append(np.array(all_s1))
                atom_masks.append(atom_mask)
                idx += 1
                if idx != nshots:
                    clear_output(wait=True)
        out_S = np.array(out_S)
        out_S2 = np.array(out_S2)
        atom_masks = np.array(atom_masks)

        return out_S, out_S2, out_s1, ts, atom_masks

    def _savedata(self, savedir, saveprefix, data, args, kwargs):
        if savedir is not None and saveprefix is not None:
            if not os.path.isdir(savedir):
                print("Making directory", savedir)
                os.mkdir(savedir)
            fname = saveprefix + "." + str(datetime.datetime.now()).replace(":", ".")
            pickle.dump({"data": data, "args": args, "kwargs": kwargs},
                    open(os.path.join(savedir, fname + ".pickle"), "wb"))

    def _get_J(self, Nx, Ny, dt):
        Nlattice = Nx * Ny
        N = np.sum(self.curr_atom_mask)

        J0 = 0.542  # Hz, nearest-neighbor
        interacs = np.array([J0, J0, 0])  # in Hz for f (not omega)
        Jint = 2 * np.pi * np.diag(interacs).reshape((1, 1, 3, 3))
        
        # Do tunneling
        for di in [1, Ny]:
            for i in range(N):
                tunnel_locs = np.array([
                    self.atom_locs[i]-di, self.atom_locs[i]+di
                ])
                tunnel_locs = tunnel_locs[(tunnel_locs >= 0) & (tunnel_locs < Nlattice)]
                tunnel_locs = tunnel_locs[self.curr_atom_mask[tunnel_locs] == False]
                if len(tunnel_locs) > 0:
                    j = self.atom_locs[i]

                    if self.tunnel_type == "simple":
                        if np.random.random() < 4*self.tunnel_freq*dt:
                            j = tunnel_locs[np.random.randint(0, len(tunnel_locs))]
                    elif self.tunnel_type == "metropolis-hastings":
                        total_propose = 4*len(tunnel_locs)*self.tunnel_freq*dt
                        if total_propose > 1:
                            print("WARNING: dt is too long for Metropolis-Hastings tunneling")
                        rv = np.random.random() 
                        if rv < total_propose:
                            pj = int(rv/(4*self.tunnel_freq*dt))
                            A = 1.0
                            if self.tunnel_P[self.atom_locs[i]] > 0:
                                A = min(1.0, self.tunnel_P[tunnel_locs[pj]]/self.tunnel_P[self.atom_locs[i]])
                            if np.random.random() <= A:
                                j = tunnel_locs[pj]
                    else:
                        print("ERROR: invalid tunnel type '%s'" % self.tunnel_type)

                    self.curr_atom_mask[self.atom_locs[i]] = False
                    self.curr_atom_mask[j] = True
                    self.atom_locs[i] = j
        
        all_ind = np.indices((Nx, Ny)).reshape((2, Nlattice))
        ind = np.array([ all_ind[:,atom] for atom in self.atom_locs ]).T

        dists = np.linalg.norm(ind.reshape((2, N, 1)) - ind.reshape((2, 1, N)), axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            J = Jint / (dists ** 3).reshape((N, N, 1, 1))
        J[np.isnan(J) | np.isinf(J)] = 0

        if self.avg_all_Js:
            J[J != 0] = np.mean(J[J != 0])
            print("J[0,1]:", J[0,1])

        return J
            
    def _run_dtwa(self, occupations, endt, nt, disorder, retallsites, pulses, tsteps, verbose, savedir, saveprefix, analysis_mask, diff_mask, s0, debugfunc):
            usetf = True
            precision = 32
            retallsites2 = False
            retallts = False
            corr = False
            userot = False

            # XX, YY, ZZ interactions
            J0 = 0.543  # Hz, nearest-neighbor
            interacs = np.array([J0, J0, 0])  # in Hz for f (not omega)

            s0 = 0.5 * s0 / np.linalg.norm(s0)
            print("s0:", s0)

            pulses_copy = pulses.copy()

            ts = []
            all_S = []
            all_S2 = []
            all_s1 = []
            all_s2 = []

            ts = np.linspace(0, endt, tsteps + 1)[1:]
            
            Jint = 2 * np.pi * np.diag(interacs).reshape((1, 1, 3, 3))
            Nx, Ny = occupations.shape
            Nlattice = Nx * Ny

            initial_atom_mask = (np.random.rand(Nlattice) < occupations.flatten())
            self.atom_locs = []
            for i in range(Nlattice):
                if initial_atom_mask[i]:
                    self.atom_locs.append(i)
            self.atom_locs = np.array(self.atom_locs)
            self.curr_atom_mask = initial_atom_mask.copy()
            
            dt = ts[1] - ts[0]
            
            def stepfunc(ti, dt, s1, s2, J, usetf=False):
                if debugfunc is not None:
                    debugfunc(ti)

                nonlocal pulses
                t = dt * (ti + 1)

                if retallsites:
                    S = np.sum(s1, axis=0)
                else:
                    S = s1
                if retallsites2:
                    S2 = np.sum(s2, axis=(0,1))
                else:
                    S2 = s2

                all_S.append(S)
                all_S2.append(S2)
                if retallsites:
                    all_s1.append(s1)
                if retallsites2:
                    all_s2.append(s2)

                ts.append(t)
               
                if self.tunnel_freq > 0: 
                  return { "J": self._get_J(Nx, Ny, dt) }

                if len(pulses) > 0 and t > pulses[0][0]:
                    pulse = Pulse(*pulses[0][1:])
                    pulses = pulses[1:]
                    print("Applying pulse:", pulse)
                    return { "rot": pulse }

            N = np.sum(initial_atom_mask)
            fill_frac = N / Nlattice

            print("N:", N, "  fill_frac:", fill_frac, flush=True)

            J = self._get_J(Nx, Ny, dt)

            R = np.zeros((N, 3))
            if disorder is not None and self.tunnel_freq == 0:
                R[:,2] = 2 * np.pi * disorder.flatten()[initial_atom_mask]
                
            analysis_mask_shot = np.array(analysis_mask).flatten()[initial_atom_mask] if analysis_mask is not None else None
            print("N in analysis_mask_shot:", np.sum(analysis_mask_shot))
            
            diff_mask_shot = np.array(diff_mask).flatten()[initial_atom_mask] if analysis_mask is not None and diff_mask is not None else None
            print("N in diff_mask_shot:", np.sum(diff_mask_shot))

            ts = []
            all_S = []
            all_S2 = []
            all_s1 = []
            #  with tf.device():
            args = (s0, J, R)
            kwargs = {"t": endt, "tsteps": tsteps, "nt": nt, "usetf": usetf, "corr": corr, \
                      "verbose": verbose, "userot": userot, "retallts": retallts, "precision": precision,
                      "retallsites": retallsites, "retallsites2": retallsites2, "analysis_mask": analysis_mask_shot, "diff_mask": diff_mask_shot }
            if usetf:
                with tf.device("/GPU:0"):
                    dtwa_lib.dtwa(*args, **kwargs, stepfunc=stepfunc)
            else:
                dtwa_lib.dtwa(*args, **kwargs, stepfunc=stepfunc)

            save_args = (occupations, endt, savedir, saveprefix, s0, disorder, pulses_copy, J0)
            self._savedata(savedir, saveprefix,
                     {"ts": ts, "all_S": all_S, "all_S2": all_S2, "all_s1": all_s1, "atom_mask": initial_atom_mask}, save_args, kwargs)

            return all_S, all_S2, all_s1, ts, initial_atom_mask
