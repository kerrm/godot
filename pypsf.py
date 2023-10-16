"""
A module to manage the PSF from CALDB and handle the integration over
incidence angle and intepolation in energy required for the binned
spectral analysis.
$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like/pypsf.py,v 1.39 2017/08/16 19:57:07 burnett Exp $
author: M. Kerr

"""

from astropy.io import fits
import numpy as np

import pycaldb

def psf_base(g,s,delta):
    """Implement the PSF base function; g = gamma, s = sigma (scaled), 
       delta = deviation in radians.
       
        Operation is vectorized both in parameters and angles; return
        is an array of shape (n_params,n_angles)."""
    u = 0.5 * np.outer(delta,1./s)**2
    return (1-1./g)*(1+u/g)**(-g)

def psf_base_integral(g,s,dmax,dmin=0):
    """Integral of the PSF base function; g = gamma, s = sigma (scaled),
       delta = deviation in radians."""
    u0 = 0.5 * np.outer(dmin,1./s)**2
    u1 = 0.5 * np.outer(dmax,1./s)**2
    return (1+u0/g)**(1-g)-(1+u1/g)**(1-g)

class PSFScaleFunc():
    """ Functor implementing the PSF energy pre-scaling."""

    def __init__(self,psf_scaling_params):
        self.c0 = psf_scaling_params[0]
        self.c1_2 = psf_scaling_params[1]**2
        self.idx = psf_scaling_params[-1]

    def __call__(self,e):
        return ((self.c0*(e*(1./100))**self.idx)**2 + self.c1_2)**0.5

class CALDBPsf(object):

    def __init__(self,irf='P8R3_SOURCE_V3'):
        self._readCALDB(irf)

    def _readCALDB(self,irf):

        man = pycaldb.CALDBManager(irf)

        def tab(dat):
            psfp = ['NCORE','NTAIL','GCORE','GTAIL','SCORE','STAIL']
            ne = dat['energ_lo'].shape[-1]
            nc = dat['ctheta_lo'].shape[-1]
            tabs = np.asarray(
                    [np.reshape(dat[p],[nc,ne]).transpose() for p in psfp])
            ebounds = np.squeeze(dat['energ_lo']),np.squeeze(dat['energ_hi'])
            cbounds = np.squeeze(dat['ctheta_lo']),np.squeeze(dat['ctheta_hi'])
            return ebounds,cbounds,np.asarray(tabs)

        hdus = [fits.open(x) for x in man.get_psf()]
        self._scale_funcs = dict()
        self._psf_params = dict()
        self._ebounds = dict()
        self._cbounds = dict()
        for hdu in hdus:
            for table in hdu:
                if table.name.startswith('PSF_SCALING_PARAMS'):
                    event_type = table.name.split('_')[-1]
                    val = np.squeeze(table.data.field('PSFSCALE'))
                    sf = np.squeeze(val)
                    self._scale_funcs[event_type] = PSFScaleFunc(sf)
                if table.name.startswith('RPSF'):
                    event_type = table.name.split('_')[-1]
                    dat = table.data
                    ebounds,cbounds,psf_params = tab(dat)
                    self._ebounds[event_type] = ebounds
                    self._cbounds[event_type] = cbounds
                    self._psf_params[event_type] = psf_params
            hdu.close()

        # apply normalization convention to PSF parameters
        # recall table shape is N_CT X N_PAR X N_EN X N_INC
        for event_type in self._psf_params.keys():

            # normalize according to the irfs/latResponse conventions
            ens = self.ecens(event_type)
            scale_func = self._scale_funcs[event_type]
            psf_params = self._psf_params[event_type]
            assert(len(ens)==psf_params.shape[1])
            for i in range(len(ens)): # iterate through energy
                sf = scale_func(ens[i])
                # vector operations in incidence angle
                nc,nt,gc,gt,sc,st = psf_params[:,i,:]
                normc = psf_base_integral(gc,sc*sf,np.pi/2)[0]
                normt = psf_base_integral(gt,st*sf,np.pi/2)[0]
                # NB leave scale factor out here so we can adjust norm to
                # a particular energy (effectively cancelling in integral)
                norm = ((2*np.pi)*(normc*sc**2+normt*nt*st**2))**-1
                self._psf_params[event_type][0,i,:] = norm # adjust NCORE
                self._psf_params[event_type][1,i,:]*= norm # --> NTAIL*NCORE

    def ecens(self,event_type):
        elo,ehi = self._ebounds[event_type]
        return (elo*ehi)**0.5

    def ccens(self,event_type):
        clo,chi = self._cbounds[event_type]
        return 0.5*(clo+chi)

    def event_types(self):
        return self._psf_params.keys()

    def get_p(self,e,event_type):
        elo,ehi = self._ebounds[event_type]
        idx = min(np.searchsorted(ehi,e),len(ehi) - 1)
        return self._psf_params[event_type][:,idx,:]

    def __call__(self,e, event_type, delta, scale_sigma=True, density=True):
        """ Return the psf density at given energy and offset.

        NB this is always in units of photons / steradian.

        Parameters
        ----------
        e : energy (MeV)
        event_type : e.g. FRONT, BACK, PSF0, ..., PSF3
        delta : offset from PSF center (radians)
        scale_sigma : apply energy pre-scaling
        density : divide by "jacobian" to turn into photon density

        Returns
        -------
        The PSF density in each cosine theta bin
        """
        sf = self._scale_funcs[event_type](e) if scale_sigma else 1
        nc,nt,gc,gt,sc,st = self.get_p(e,event_type)
        yc = psf_base(gc,sc*sf,delta)
        yt = psf_base(gt,st*sf,delta)
        return (nc*yc + nt*yt)/sf**2

    def integral(self,e,event_type,dmax,dmin=0):
        """ Return integral of PSF at given energy and conversion type,
            from dmin to dmax (radians).

            Parameters
            ----------
            e : energy in MeV
            event_type : e.g. FRONT, BACK, PSF0, ..., PSF3
            dmax : integral upper bound (radians)

            Returns
            -------
            The integral in each cosine theta bin.
        """
        nc,nt,gc,gt,sc,st = self.get_p(e,event_type)
        sf = self._scale_funcs[event_type](e)
        icore = (2*np.pi)*sc**2*nc*\
                psf_base_integral(gc,sc*sf,dmax,dmin=dmin)[0]
        itail = (2*np.pi)*st**2*nt*\
                psf_base_integral(gt,st*sf,dmax,dmin=dmin)[0]
        return icore+itail

    def inverse_integral(self,e,event_type,percent=68,on_axis=False):
        """Determine radius at which integral PSF contains specified pctg.

        Parameters
        ----------
        e : energy in MeV
        event_type : e.g. FRONT, BACK, PSF0, ..., PSF3

        Returns
        -------
        val : PSF radius in degrees
        """
        percent = float(percent)/100
        if on_axis:
            sf = self._scale_funcs[event_type](e)
            nc,nt,gc,gt,sc,st,w = self.get_p(e,event_type)
            u1 = gc[0]*( (1-percent)**(1./(1-gc[0])) - 1)
            u2 = gt[0]*( (1-percent)**(1./(1-gt[0])) - 1)
            return np.degrees(sf*(nc[0]*(u1*2)**0.5*sc[0] + nt[0]*(u2*2)**0.5*st[0])) #approx
        f = lambda x: abs(self.integral(e,ct,x) - percent)
        from scipy.optimize import fmin
        seeds = np.asarray([5,4,3,2.5,2,1.5,1,0.5,0.25])*self._scale_funcs[event_type](e)
        seedvals = np.asarray([self.integral(e,ct,x) for x in seeds])
        seed = seeds[np.argmin(np.abs(seedvals-percent))]
        trial = fmin(f,seed,disp=0,ftol=0.000001,xtol=0.01)
        if trial > 0:
            return np.degrees(trial[0])
        print('Warning: could not invert integral; return best grid value.')
        return np.degrees(seed)

