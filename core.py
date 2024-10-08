from collections import deque

from astropy.io import fits
import numpy as np
import pylab as pl
from scipy.integrate import simps,cumtrapz
from scipy.optimize import fmin,fsolve,fmin_tnc,brentq
from scipy.interpolate import interp1d
from scipy.stats import chi2

from . import bary
from . import events
from . import py_exposure_p8

from importlib import reload
reload(py_exposure_p8)

dbug = dict()

# MET bounds for 8-year data set used for FL8Y and 4FGL
t0_8year = 239557007.6
t1_8year = 491999980.6

def met2mjd(times,mjdref=51910+7.428703703703703e-4):
    times = np.asarray(times,dtype=np.float128)
    return times*(1./86400)+mjdref

def mjd2met(times,mjdref=51910+7.428703703703703e-4):
    times = (np.asarray(times,dtype=np.float128)-mjdref)*86400
    return times

def infer_met(time):
    if time < 100000:
        return mjd2met(time)
    return time

def calc_weighted_aeff(pcosines,phi,base_spectrum=None,
        ltfrac=None,correct_psf=True,
        use_psf_types=True,type_selector=None,emin=100,emax=1e5,
        verbosity=2):
    """ Compute the "effective effective area" for a set of exposure
    segments, typically the ~30s intervals tabulated in an FT2 files.

    Typically, this will involve averaging the effective area over the
    energy range.  Additionally, by default, the finite aperture size
    will be taken into account by computing the fraction of the PSF
    contained at each energy slice.

    Parameters
    ----------
    pcosines : the polar angle cosines, vz. cos(theta)
    phi : azimuthal angle (radians); if not None, then the azimuthal
        dependence of the effective area will be applied
    base_spectrum : a function returning dN/dE(E) for the source; if None,
        will use an E^-2 weighting
    ltfrac : a livetime fraction entry for each interval. If not None,
        will apply the efficiency correction to the aeff
    correct_psf : scale the effective area by the fraction of the PSF
        contained at the energy slice
    type_selector : an instance of PSFTypeSelector, e.g., which will
        give an energy-dependent event type selection.  Those events
        which are not selected will be omitted from the effective area
        sum.
    use_psf_types : event types are PSF types; otherwise, front/back
    emin : minimum energy of integration (MeV)
    emax : maximum energy of integration (MeV)
    """

    emin = np.log10(emin)
    emax = np.log10(emax)
    ea = py_exposure_p8.EffectiveArea()
    if correct_psf:
        pc = py_exposure_p8.PSFCorrection()
    else:
        pc = None
    if ltfrac is not None:
        ec = py_exposure_p8.EfficiencyCorrection()
    else:
        ec = None
    if use_psf_types:
        event_types = [f'PSF{i}' for i in range(4)]
        event_codes = range(0,4)
    else:
        event_types = ['FRONT','BACK']
        event_codes = range(0,2)

    if base_spectrum is None:
        base_spectrum = lambda E: E**-2

    # try to pick something like 8/decade
    nbin = int(round((emax-emin)*8))+1
    edom = np.logspace(emin,emax,nbin)
    wts = base_spectrum(edom)
    # this is the exposure as a function of energy
    total_exposure = np.empty_like(edom)
    rvals = np.zeros([len(edom),len(pcosines)])
    for i,(en,wt) in enumerate(zip(edom,wts)):
        for etype,ecode in zip(event_types,event_codes):
            if type_selector is not None:
                # check to see if we are using this en/ct
                if not type_selector.accept(en,ecode):
                    if verbosity >= 2:
                        print(f'Skipping {en:.2f},{ecode}.')
                    continue
            aeff = ea(en,pcosines,etype,phi=phi)
            if pc is not None: # PSF correction
                aeff *= pc(en,pcosines,etype)
            if ec is not None: # efficiency correction
                aeff *= ec(en,ltfrac,etype)
            rvals[i] += aeff
        total_exposure[i] = rvals[i].sum()
        rvals[i] *= wt
    aeff = simps(rvals,edom,axis=0)/simps(wts,edom)

    return aeff,edom,total_exposure


class PSFTypeSelector():
    """ Select event types as a function of energy.  This is used for
        selecting events and for computing the weighted effective area.

    We assume there is no maximum energy for a given PSF type.
    """
    def __init__(self,min_energies=10**np.asarray([2.75,2.5,2.25,2.00])):
        if not len(min_energies)==4:
            raise ValueError('Provide a minimum energy for each PSF type.')
        self._men = np.asarray(min_energies)
        if not np.all((self._men[:-1] > self._men[1:])):
            raise Warning('Minimum energies should probably decrease with PSF type.')

    def accept(self,en,evtype):
        """ Return TRUE where the evtype satisfies the minimum for the
            provided energy.
        """
        en = np.atleast_1d(en)
        evtype = np.atleast_1d(evtype)
        ok = np.full(len(en),False,dtype=bool)
        for i in range(4):
            m = evtype==i
            ok[m] = en[m] >= self._men[i]
        return np.squeeze(ok)


class FBTypeSelector():
    """ Select event types as a function of energy.  This is used for
        selecting events and for computing the weighted effective area.

    It is assumed there is no minimum energy for front-type events, so
    the only parameter is the minimum energy for BACK events.
    """
    def __init__(self,min_back_energy=10**2.5):
        self._mbe = min_back_energy

    def accept(self,en,evtype):
        """ Return TRUE where the evtype satisfies the minimum for the
            provided energy.
        """
        return evtype==0 | (evtype==1 & en >= self._mbe)


class Cell(object):
    """ Encapsulate the concept of a Cell, specifically a start/stop
    time, the exposure, and set of photons contained within."""

    def __init__(self,tstart,tstop,exposure,photon_times,photon_weights,
            source_to_background_ratio):
        self.tstart = tstart
        self.tstop = tstop
        self.exp = exposure
        self.ti = np.atleast_1d(photon_times)
        self.we = np.atleast_1d(photon_weights)
        self.SonB = source_to_background_ratio
        # TMP? Try baking in a set of scales to allow re-scaling of a
        # baseline set of weights.  Ultimately we need this at the data
        # level too...
        self._alpha = 1
        self._beta = 1

    def sanity_check(self):
        if self.exp==0:
            assert(len(self.ti)==0)
        assert(len(self.ti)==len(self.we))
        assert(np.all(self.ti >= self.tstart))
        assert(np.all(self.ti < self.tstop))

    def get_tmid(self):
        return 0.5*(self.tstart+self.tstop)

class PhaseCell(Cell):

    def __init__(self,tstart,tstop,exposure,photon_times,photon_weights,
            source_to_background_ratio):
        super(PhaseCell,self).__init__(tstart,tstop,exposure,photon_times,photon_weights,source_to_background_ratio)

    def copy(self,phase_offset=0):
        return PhaseCell(self.tstart+phase_offset,self.tstop+phase_offset,
                self.exp,self.ti,self.we,self.SonB)
        

def cell_from_cells(cells):
    """ Return a single Cell object for multiple cells."""

    cells = sorted(cells,key=lambda cell:cell.tstart)
    tstart = cells[0].tstart
    we = np.concatenate([c.we for c in cells])
    ti = np.concatenate([c.ti for c in cells])
    exp = np.asarray([c.exp for c in cells]).sum()
    bexp = np.asarray([c.exp/c.SonB for c in cells]).sum()
    SonB = exp/bexp
    return Cell(cells[0].tstart,cells[-1].tstop,exp,ti,we,SonB)

class CellTimeSeries(object):
    """ Encapsulate binned data from cells, specifically the exposure,
    the cell edges, and the first three moments of the photon weights
    (counts, weights, weights^2 in each cell.)

    The purpose of this class/time series is a lightweight version of
    the cell data for use in uniform sampling algorithms like FFTs.

    It's possible that during rebinning, some small-exposure cells will
    come to be.  A minimum exposure cut here keeps numerical problems at
    bay.
    """

    def __init__(self,starts,stops,exp,sexp,bexp,
            counts,weights,weights2,deadtime,
            alt_starts=None,alt_stops=None,timesys='barycenter',
            minimum_exposure=3e4):
        self.starts = starts
        self.stops = stops
        self.exp = exp
        self.sexp = sexp
        self.bexp = bexp
        self.counts = counts
        self.weights = weights
        self.weights2 = weights2
        self.deadtime = deadtime
        self.alt_starts = alt_starts
        self.alt_stops = alt_stops
        self.timesys = timesys
        ## exposure mask
        self.minimum_exposure = minimum_exposure
        mask = ~(exp > minimum_exposure)
        self._zero_weight(mask)

    def save(self,fname,compress=True):
        keys = ['starts','stops','exp','sexp','bexp','counts','weights',
                'weights2','deadtime']
        if self.alt_starts is not None:
            keys += ['alt_starts']
        if self.alt_stops is not None:
            keys += ['alt_stops']
        d = dict()
        for key in keys:
            d[key] = getattr(self,key)
        d['metadata'] = [str(self.minimum_exposure),self.timesys]
        if compress:
            np.savez_compressed(fname,**d)
        else:
            np.savez(fname,**d)

    @staticmethod
    def load(fname):
        q = np.load(fname)
        minimum_exposure = float(q['metadata'][0])
        timesys = q['metadata'][1]
        alt_starts = alt_stops = None
        if 'alt_starts' in q.keys():
            alt_starts = q['alt_starts']
        if 'alt_stops' in q.keys():
            alt_stops = q['alt_stops']
        keys = ['starts','stops','exp','sexp','bexp','counts','weights',
                'weights2','deadtime']
        args = [q[key] for key in keys]
        kwargs = dict(alt_starts=alt_starts,alt_stops=alt_stops,timesys=timesys,minimum_exposure=minimum_exposure)
        q.close()
        return CellTimeSeries(*args,**kwargs)

    def _zero_weight(self,mask):
        self.exp[mask] = 0
        self.sexp[mask] = 0
        self.bexp[mask] = 0
        self.counts[mask] = 0
        self.weights[mask] = 0
        self.weights2[mask] = 0

    def tsamp(self):
        # TODO -- perhaps put in a contiguity check
        return self.stops[0]-self.starts[0]

    def ps_nbin(self):
        """ Return the expected number of frequency bins in the likelihood
            PSD, with no zero padding.
        """
        return len(self.starts)//4 + 1

    def tspan(self):
        return self.stops[-1]-self.starts[0]

    def get_topo_bins(self):
        if self.timesys=='barycenter':
            if self.alt_starts is None:
                raise Exception('Do not have topocentric bins.')
            return self.alt_starts,self.alt_stops
        else:
            return self.starts,self.stops

    def get_bary_bins(self):
        if self.timesys!='barycenter':
            if self.alt_starts is None:
                raise Exception('Do not have barycentric bins.')
            return self.alt_starts,self.alt_stops
        else:
            return self.starts,self.stops

    def zero_weight(self,tstart,tstop):
        """ Zero out data between tstart and tstop.

        Parameters
        ----------
        tstart : beginning of zero interval (MET, but will infer from MJD)
        ttop : end of zero interval (MET, but will infer from MJD)
        """
        tstart = infer_met(tstart)
        tstop = infer_met(tstop)
        mask = (self.starts >= tstart) & (self.stops <= tstop)
        self._zero_weight(mask)

class CellLogLikelihood(object):

    def __init__(self,cell,swap=False):
        """ 
        Parameters
        ----------
        cell : a Cell object to form the likelihood for
        swap : swap the source and the background
        """
        self.cell = cell
        self.ti = cell.ti
        #self.we = cell.we.astype(np.float64)
        a,b,w = cell._alpha,cell._beta,cell.we.astype(np.float64)
        self.we = w*a / (w*a + b*(1-w))
        self.iwe = 1-self.we
        self.S = cell.exp*cell._alpha
        self.B = cell.exp/cell.SonB*cell._beta
        if swap:
            self.we,self.iwe = self.iwe,self.we
            self.S,self.B = self.B,self.S
        self._tmp1 = np.empty_like(self.we)
        self._tmp2 = np.empty_like(self.we)
        self._last_beta = 0.

    def log_likelihood(self,alpha):
        """ Return the (positive) log likelihood for a flux multiplier
            alpha.  Viz, the flux in this cell is alpha*F_mean, so the
            null hypothesis is alpha=1.
        """
        # NB the minimum defined alpha is between -1 and 0 according to
        # amin = (wmax-1)/wmax
        t1,t2 = self._tmp1,self._tmp2
        np.multiply(self.we,alpha,out=t1)
        np.add(t1,self.iwe,out=t1)
        np.log(t1,out=t2)
        return np.sum(t2)-(self.S*alpha)

    def log_profile_likelihood_approx(self,alpha):
        """ Profile over the background level under assumption that the
        background variations are small (few percent).
        
        For this formulation, we're using zero-based parameters, i.e
        flux = F0*(1+alpha)."""

        alpha = alpha-1
        t1,t2 = self._tmp1,self._tmp2
        np.multiply(self.we,alpha,out=t1)
        t1 += 1.
        # t1 is now 1+alpha*we
        np.divide(self.iwe,t1,out=t2)
        #t2 is now 1-w/1+alpha*w
        Q = np.sum(t2)
        t2 *= t2
        R = np.sum(t2)
        # for reference
        beta_hat = (Q-self.B)/R

        t1 += beta_hat*(1-self.we)
        np.log(t1,out=t2)
        #return np.sum(t2) + 0.5*(Q-B)**2/R -alpha*self.exp
        return np.sum(t2) - (1+alpha)*self.S - (1+beta_hat)*self.B

    def log_profile_likelihood(self,alpha,beta_guess=1):
        """ Profile over the background level with no restriction on
        amplitude.  Use a prescription similar to finding max on alpha.

        For this formulation, we're using zero-based parameters.
        """

        beta = self._last_beta = self.get_beta_max(
                alpha,guess=beta_guess)-1
        alpha = alpha-1
        t1,t2 = self._tmp1,self._tmp2
        np.multiply(self.we,alpha-beta,out=t1)
        np.add(1.+beta,t1,out=t1)
        # t1 is now 1+beta+we(alpha-beta)
        np.log(t1,out=t1)
        return np.sum(t1)-self.S*(1+alpha)-self.B*(1+beta)

    def get_likelihood_grid(self,amax=2,bmax=2,res=0.01):
        na = int(round(amax/res))+1
        nb = int(round(bmax/res))+1
        agrid = np.linspace(0,amax,na)-1
        bgrid = np.linspace(0,bmax,nb)-1
        rvals = np.empty((na,nb))
        S,B = self.S,self.B
        t1,t2 = self._tmp1,self._tmp2
        iw = self.iwe
        for ia,alpha in enumerate(agrid):
            t2[:] = alpha*self.we+1
            for ib,beta in enumerate(bgrid):
                np.multiply(beta,iw,out=t1)
                np.add(t2,t1,out=t1)
                np.log(t1,out=t1)
                rvals[ia,ib] = np.sum(t1) -B*(1+beta)
            rvals[ia,:] -= S*(1+alpha) 
        return agrid+1,bgrid+1,rvals

    def log_full_likelihood(self,p):
        """ Likelihood for both source and background normalizations.
        """

        alpha,beta = p
        alpha -= 1
        beta -= 1
        t1,t2 = self._tmp1,self._tmp2
        np.multiply(alpha-beta,self.we,out=t1)
        np.add(1+beta,t1,out=t1)
        np.log(t1,out=t1)
        return np.sum(t1) - self.S*(1+alpha) -self.B*(1+beta)

    def fmin_tnc_func(self,p):
        alpha,beta = p
        alpha = p[0] - 1
        beta = p[1] -1
        if (alpha==-1) and (beta==-1):
            # would call log(0)
            return np.inf,[np.inf,np.inf]
        S,B = self.S,self.B
        t1 = np.multiply(self.we,alpha-beta,out=self._tmp1)
        t1 += 1+beta
        if np.any(t1 <= 0):
            print(alpha,beta,self.we[t1 <= 0])
        # above equivalent to
        # t1[:] = 1+beta*iw+alpha*self.we
        t2 = np.log(t1,out=self._tmp2)
        logl = np.sum(t2) - S*alpha -B*beta
        np.divide(self.we,t1,out=t2)
        grad_alpha = np.sum(t2)-S
        np.divide(self.iwe,t1,out=t2)
        grad_beta = np.sum(t2)-B
        return -logl,[-grad_alpha,-grad_beta]

    def fmin_fsolve(self,p):
        alpha,beta = p
        alpha -= 1
        beta -= 1
        S,B = self.S,self.B
        w = self.we
        iw = 1-self.we
        t1,t2 = self._tmp1,self._tmp2
        t1[:] = 1+beta*iw+alpha*w
        grad_alpha = np.sum(w/t1)-S
        grad_beta = np.sum(iw/t1)-B
        print(p,grad_alpha,grad_beta)
        return [-grad_alpha,-grad_beta]

    def fmin_fsolve_jac(self,p):
        pass

    def f1(self,alpha):
        w = self.we
        return np.sum(w/(alpha*w+(1-w)))-self.S

    def f1_profile(self,alpha):
        w = self.we
        a = alpha-1
        S,B = self.S,self.B
        t1 = self._tmp1
        t2 = self._tmp2
        t1[:] = (1-w)/(1+a*w)
        np.multiply(t1,t1,out=t2)
        t3 = w/(1+a*w)
        T2 = np.sum(t2)
        beta_hat = (np.sum(t1)-B)/T2
        t1_prime = np.sum(t3*t1)
        t2_prime = np.sum(t3*t2)
        beta_hat_prime = (2*beta_hat*t2_prime-t1_prime)/T2
        return np.sum((w+beta_hat_prime*(1-w))/(1+a*w+beta_hat*(1-w))) -S -beta_hat_prime*B

    def f2(self,alpha):
        w = self.we
        t = alpha*w+(1-w)
        return -np.sum((w/t)**2)

    def f3(self,alpha):
        w = self.we
        t = alpha*w+(1-w)
        return np.sum((w/t)**3)

    def nr(self,guess=1,niter=6):
        """ Newton-Raphson solution to max."""
        # can precalculate alpha=0, alpha=1 for guess, but varies depending
        # on TS, might as well just stick to 1 and use full iteration
        a = guess
        w = self.we
        iw = 1-self.we
        S = self.S
        t = np.empty_like(self.we)
        for i in range(niter):
            t[:] = w/(a*w+iw)
            f1 = np.sum(t)-S
            t *= t
            f2 = -np.sum(t)
            a = max(0,a - f1/f2)
        return a

    def halley(self,guess=1,niter=5):
        """ Hally's method solution to max."""
        a = guess
        w = self.we
        iw = 1-self.we
        S = self.S
        t = np.empty_like(self.we)
        t2 = np.empty_like(self.we)
        for i in range(niter):
            t[:] = w/(a*w+iw)
            f1 = np.sum(t)-S
            np.multiply(t,t,out=t2)
            f2 = -np.sum(t2)
            np.multiply(t2,t,out=t)
            f3 = 2*np.sum(t)
            a = max(0,a - 2*f1*f2/(2*f2*f2-f1*f3))
        return a

    def get_max(self,guess=1,beta=1,profile_background=False,
            recursion_count=0):
        """ Find value of alpha that optimizes the log likelihood.

        Is now switched to the 0-based parameter convention.

        Use an empirically tuned series of root finding.
        """
        if profile_background:

            rvals,nfeval,rc = fmin_tnc(self.fmin_tnc_func,
                    [float(guess),float(beta)],
                    bounds=[[0,None],[0,None]],disp=0,ftol=1e-3,
                    maxfun=200)

            if not((rc < 0) or (rc > 2)):
                if (guess == 0) and (rvals[0] > 5e-2):
                    print('Warning, possible inconsistency.  Guess was 0, best fit value %.5g.'%(rvals[0]),'beta=',beta)
                return rvals

            # just do a second iteration to try see if it converges
            guess = rvals[0]
            beta = rvals[1]
            guess += 1e-3 # just perturb these a bit
            beta -= 1e-3

            rvals,nfeval,rc = fmin_tnc(self.fmin_tnc_func,
                    [guess,beta],
                    bounds=[[0,None],[0,None]],disp=0,ftol=1e-3,
                    maxfun=200)

            if not((rc < 0) or (rc > 2)):
                if (guess == 0) and (rvals[0] > 5e-2):
                    print('Warning, possible inconsistency.  Guess was 0, best fit value %.5g.'%(rvals[0]),'beta=',beta)
                return rvals
            
            oldguess = guess
            oldbeta = beta
            oldrvals = rvals
            oldlogl = self.fmin_tnc_func(rvals)[0]

            if rc == 3: # exceeded maximum evaluations
                newguess = oldguess
                newbeta = oldbeta
            else:
                # try a small grid to seed a search
                grid = np.asarray([0,0.1,0.3,0.5,1.0,1.5,2.0,5.0,10.0])
                cogrid = np.asarray(
                        [self.log_profile_likelihood(x) for x in grid])

                idx_amax = np.argmax(cogrid)
                idx_less = max(0,idx_amax-1)
                idx_gret = min(idx_amax+1,len(grid)-1)
                dlogl = abs(cogrid[idx_gret]-cogrid[idx_less])
                if dlogl > 100:
                    # make a finer grid
                    grid = np.linspace(grid[idx_less],grid[idx_gret],20)
                    cogrid = np.asarray(
                            [self.log_profile_likelihood(x) for x in grid])

                # need to cast these otherwise the TNC wrapper barfs on a
                # type check...
                newguess = float(grid[np.argmax(cogrid)])
                newbeta = float(max(0.1,self.get_beta_max(newguess)))

            rvals,nfeval,rc = fmin_tnc(self.fmin_tnc_func,
                    [newguess,newbeta],
                    bounds=[[0,None],[0,None]],disp=0,ftol=1e-3,
                    maxfun=200)
            newrvals = rvals
            newlogl = self.fmin_tnc_func(rvals)[0]

            if not((rc < 0) or (rc > 2)):
                if newlogl > oldlogl:
                    if newlogl > (oldlogl+2e-3):
                        print(f'Warning! Converged but log likelihood decreased; rc={rc}.')
                    return oldrvals
                return newrvals
            else:
                # alpha = 0, but the log likelihoods agree, so don't emit
                # a warning
                if (rvals[0] == 0) and np.all((cogrid[1:]-cogrid[:-1]<0)):
                    pass
                else:
                    print('Warning, never converged locating maximum with profile_background!  Results for this interval may be unreliable.')

            return newrvals

        w = self.we
        iw = self.iwe
        S,B = self.S,self.B
        guess = guess-1
        beta = beta-1

        # check that the maximum isn't at flux=0 (alpha-1) with derivative
        a = -1
        t1,t2 = self._tmp1,self._tmp2
        t2[:] = 1+beta*iw
        t1[:] = w/(t2+a*w)
        if (np.sum(t1)-S) < 0:
            return 0
        else:
            a = guess

        # on first iteration, don't let it go to 0
        t1[:] = w/(t2+a*w)
        f1 = np.sum(t1)-S
        t1 *= t1
        f2 = np.sum(t1) # will include sign below
        a = a + f1/f2
        if a < 0-1:
            a = 0.2-1

        # second iteration more relaxed
        t1[:] = w/(t2+a*w)
        f1 = np.sum(t1)-S
        t1 *= t1
        f2 = np.sum(t1) # will include sign below
        a = a + f1/f2
        if a < 0.05-1:
            a = 0.05-1

        # last NR iteration allow 0
        t1[:] = w/(t2+a*w)
        f1 = np.sum(t1)-S
        t1 *= t1
        f2 = np.sum(t1) # will include sign below
        alast = a = max(0-1,a + f1/f2)

        # now do a last Hally iteration
        t1[:] = w/(t2+a*w)
        f1 = np.sum(t1)-S
        t1 *= t1
        f2 = np.sum(t1) # will include sign below
        t1 *= w/(t2+a*w)
        f3 = 2*np.sum(t1)
        a = max(0-1,a + 2*f1*f2/(2*f2*f2-f1*f3))
        
        # a quick check if we are converging slowly to try again or if
        # we started very from from the guess (large value)
        if (abs(a-alast)>0.05) or (abs(guess-a) > 10):
            if recursion_count > 2:
                return self.get_max_numerical()
                #raise ValueError('Did not converge!')
            return self.get_max(guess=a+1,beta=beta+1,
                    recursion_count=recursion_count+1)

        return a+1

    def get_max_numerical(self,guess=1,beta=1,profile_background=False,
            recursion_count=0):
        """ Find value of alpha that optimizes the log likelihood.

        Is now switched to the 0-based parameter convention.

        Use an empirically tuned series of root finding.
        """
        if profile_background:
            # TODO -- probably want to replace this with a better iterative
            # method, but for now, just use good ol' fsolve!

            # test TNC method
            rvals,nfeval,rc = fmin_tnc(self.fmin_tnc_func,[guess,beta],
                    bounds=[[0,None],[0,None]],disp=0,ftol=1e-3)
            if (rc < 0) or (rc > 2):
                print('Warning, best guess probably wrong.')
            return rvals

        w = self.we
        iw = self.iwe
        S,B = self.S,self.B
        guess = guess-1
        beta = beta-1

        # check that the maximum isn't at flux=0 (alpha-1) with derivative
        a = -1
        t1,t2 = self._tmp1,self._tmp2
        t2[:] = 1+beta*iw
        t1[:] = w/(t2+a*w)
        if (np.sum(t1)-S) < 0:
            return 0
        else:
            a = guess

        def f(a):
            t1[:] = w/(t2+a*w)
            return np.sum(t1)-S

        a0 = -1
        amax = guess
        for i in range(12):
            if f(amax) < 0:
                break
            a0 = amax
            amax = 2*amax+1
        return brentq(f,a0,amax,xtol=1e-3)+1

    def get_beta_max(self,alpha,guess=1,recursion_count=0):
        """ Find value of beta that optimizes the likelihood, given alpha.
        """
        if np.isnan(alpha):
            return 1
        if len(self.we)==0:
            return 0 # maximum likelihood estimator
        if alpha == 0:
            return len(self.we)/self.B # MLE
        alpha = alpha-1
        guess -= 1
        S,B = self.S,self.B
        w = self.we
        iw = self.iwe
        t,t2 = self._tmp1,self._tmp2

        # check that the maximum isn't at 0 (-1) with derivative
        t[:] = iw/w
        if np.sum(t) < B*(1+alpha):
            return 0
        else:
            b = guess

        # on first iteration, don't let it go to 0
        t2[:] = 1+alpha*w
        t[:] = iw/(t2+b*iw)
        f1 = np.sum(t)-B
        t *= t
        f2 = np.sum(t) # will include sign below
        b = b + f1/f2
        b = max(0.2-1,b)

        # second iteration more relaxed
        t[:] = iw/(t2+b*iw)
        f1 = np.sum(t)-B
        t *= t
        f2 = np.sum(t) # will include sign below
        b = b + f1/f2
        b = max(0.05-1,b)

        # last NR iteration allow 0
        # second iteration more relaxed
        t[:] = iw/(t2+b*iw)
        f1 = np.sum(t)-B
        t *= t
        f2 = np.sum(t) # will include sign below
        b = b + f1/f2
        b = max(0.02-1,b)

        # replace last NR iteration with a Halley iteration to handle
        # values close to 0 better; however, it can result in really
        # huge values, so add a limiting step to it
        t[:] = iw/(t2+b*iw)
        f1 = np.sum(t)-B
        t *= t
        f2 = np.sum(t) # will include sign below
        t *= iw/(t2+b*iw)
        f3 = 2*np.sum(t)
        newb = max(0-1,b + 2*f1*f2/(2*f2*f2-f1*f3))
        if abs(newb-b) > 10:
            blast = b = 2*b+1
        else:
            blast = b = newb

        # now do a last Hally iteration
        t[:] = iw/(t2+b*iw)
        f1 = np.sum(t)-B
        t *= t
        f2 = np.sum(t) # will include sign below
        t *= iw/(t2+b*iw)
        f3 = 2*np.sum(t)
        b = max(0-1,b + 2*f1*f2/(2*f2*f2-f1*f3))

        # a quick check if we are converging slowly to try again or if
        # the final value is very large
        if (abs(b-blast)>0.05) or (abs(guess-b) > 10) or (b==-1):
            if recursion_count > 2:
                raise ValueError('Did not converge for alpha=%.5f!'%(
                    alpha+1))
            return self.get_beta_max(alpha+1,guess=b+1,
                    recursion_count=recursion_count+1)

        return b+1

    def get_beta_max_numerical(self,alpha,guess=1):
        alpha = alpha-1
        S,B = self.S,self.B
        w = self.we
        iw = self.iwe
        t,t2 = self._tmp1,self._tmp2

        # check that the maximum isn't at 0 (-1) with derivative
        t2[:] = 1+alpha*w
        t[:] = iw/(t2-iw)
        if (np.sum(t)-B) < 0:
            return 0

        def f(b):
            t[:] = iw/(t2+b*iw)
            return np.sum(t)-B

        b0 = -1
        bmax = guess-1
        for i in range(8):
            if f(bmax) < 0:
                break
            b0 = bmax
            bmax = 2*bmax+1
        return brentq(f,b0,bmax,xtol=1e-3)+1

    def get_logpdf(self,aopt=None,dlogl=20,npt=100,include_zero=False,
            profile_background=False):
        """ Evaluate the pdf over an adaptive range that includes the
            majority of the support.  Try to keep it to about 100 iters.
        """

        if profile_background:
            return self._get_logpdf_profile(aopt=aopt,dlogl=dlogl,npt=npt,
                    include_zero=include_zero)

        if aopt is None:
            aopt = self.get_max()
        we = self.we
        iw = self.iwe
        S,B = self.S,self.B
        t = self._tmp1
        amin = 0

        if aopt == 0:
            # find where logl has dropped, upper side
            llmax = np.log(iw).sum()
            # do a few NR iterations
            amax = max(0,-(llmax+dlogl)/(np.sum(we/(1-we))-S))
            for i in range(10):
                t[:] = amax*we+iw
                f0 = np.log(t).sum()-amax*S+dlogl-llmax
                f1 = np.sum(we/t)-S
                amax = amax - f0/f1
                if abs(f0) < 0.1:
                    break
        else:
            # find where logl has dropped, upper side
            t[:] = aopt*we + iw
            llmax = np.sum(np.log(t))-S*aopt
            # use Taylor approximation to get initial guess
            f2 = np.abs(np.sum((we/t)**2))
            amax = aopt + np.sqrt(2*dlogl/f2)
            # do a few NR iterations
            for i in range(5):
                t[:] = amax*we+iw
                f0 = np.log(t).sum()-amax*S+dlogl-llmax
                f1 = np.sum(we/t)-S
                amax = amax - f0/f1
                if abs(f0) < 0.1:
                    break

        if (not include_zero) and (aopt > 0):
            # ditto, lower side; require aopt > 0 to avoid empty we
            t[:] = aopt*we + iw
            # use Taylor approximation to get initial guess
            f2 = np.abs(np.sum((we/t)**2))
            # calculate the minimum value of a which will keep the argument
            # of the logarithm below positive, and enforce it, just to
            # avoid numerical warnings -- 25 Apr 2024
            min_a = np.max(1-1./we) + 1e-6
            amin = max(min_a,aopt - np.sqrt(2*dlogl/f2))
            # do a few Newton-Raphson iterations
            for i in range(5):
                t[:] = amin*we+iw
                f0 = np.log(t).sum()-amin*S+dlogl-llmax
                f1 = np.sum(we/t)-S
                amin = max(min_a,amin - f0/f1)
                if abs(f0) < 0.1:
                    break

        amin = max(0,amin)

        dom = np.linspace(amin,amax,npt)
        cod = np.empty_like(dom)
        for ia,a in enumerate(dom):
            cod[ia] = self.log_likelihood(a) 

        # do a sanity check here
        acodmax = np.argmax(cod)
        codmax = cod[acodmax]
        if abs(codmax - llmax) > 0.05:
            aopt = dom[acodmax]
            return self.get_logpdf(aopt=aopt,dlogl=dlogl,npt=npt,
                    include_zero=include_zero)

        cod -= llmax
        return dom,cod

    def _get_logpdf_profile(self,aopt=None,dlogl=20,npt=100,
            include_zero=False):
        """ Evaluate the pdf over an adaptive range that includes the
            majority of the support.  Try to keep it to about 100 iters.
        """
        if aopt is None:
            aopt,bopt = self.get_max(profile_background=True)
        else:
            bopt = self.get_beta_max(aopt)

        # the algebra gets pretty insane here, so I think it's easier just
        # to find the range numerically

        amin = 0
        llmax = self.log_full_likelihood([aopt,bopt])

        f = lambda a:self.log_profile_likelihood(a)-llmax+dlogl
        a0 = aopt
        amin = 0
        amax = max(5,a0)
        # make sure upper range contains root
        for i in range(4):
            if f(amax) > 0:
                a0 = amax
                amax *= amax
        amax = brentq(f,a0,amax)
        if aopt > 0:
            if f(0) > 0:
                amin = 0
            else:
                amin = brentq(f,0,aopt)

        dom = np.linspace(amin,amax,npt)
        cod = np.empty_like(dom)
        self._last_beta = 0
        for ia,a in enumerate(dom):
            cod[ia] = self.log_profile_likelihood(a)
                    #beta_guess=self._last_beta+1)

        cod -= llmax
        return dom,cod

    def get_pdf(self,aopt=None,dlogl=20,npt=100,profile_background=False):
        dom,cod = self.get_logpdf(aopt=aopt,dlogl=dlogl,npt=npt,
                profile_background=profile_background)
        np.exp(cod,out=cod)
        return dom,cod*(1./simps(cod,x=dom))

    def get_ts(self,aopt=None,profile_background=False):
        if self.S == 0:
            return 0
        if aopt is None:
            aopt = self.get_max(profile_background=profile_background)
            print(aopt)
            if profile_background:
                aopt = aopt[0] # discard beta
            print(aopt)
        if aopt == 0:
            return 0
        func = self.log_profile_likelihood if profile_background else self.log_likelihood
        return 2*(func(aopt)-func(0))

    def get_flux(self,conf=[0.05,0.95],profile_background=False):
        aopt = self.get_max(profile_background=profile_background)
        if profile_background:
            aopt = aopt[0]
        dom,cod = self.get_pdf(aopt=aopt,
                profile_background=profile_background)
        amax = np.argmax(cod)
        # Check for agreement between analytic and numerical maximum;
        # if they are out, try both to get the max with a refined guess
        # and evaluate the likelihood on a finer grid
        if abs(dom[amax]-aopt) > 0.1:
            aopt = self.get_max(guess=dom[amax],
                    profile_background=profile_background)
            if profile_background:
                aopt = aopt[0]
            dom,cod = self.get_pdf(aopt=aopt,
                    profile_background=profile_background,npt=200)
            amax = np.argmax(cod)
            # If they are still out, use the numerical value
            if abs(dom[amax]-aopt) > 0.1:
                print('failed to obtain agreement, using internal version')
                aopt = dom[amax]
        ts = self.get_ts(aopt=aopt,profile_background=profile_background)
        cdf = cumtrapz(cod,dom,initial=0)
        cdf *= 1./cdf[-1]
        indices = np.searchsorted(cdf,conf)
        # do a little linear interpolation step here
        ihi,ilo = indices,indices-1
        m = (cdf[ihi]-cdf[ilo])/(dom[ihi]-dom[ilo])
        xconf = dom[ilo] + (np.asarray(conf)-cdf[ilo])/m
        return aopt,ts,xconf

    def get_profile_flux(self):
        """ Make an expansion of the likelihood assuming a small deviation
            of the source and background density from the mean, and
            return the flux estimator after profiling over the background.

            This has not been carefully checked, but does look sane from
            some quick tests.
        """
        N = len(self.we)
        # estimate of source background from exposure
        S,B = self.S,self.B
        W = np.sum(self.we)
        W2 = np.sum(self.we**2)
        t1 = W2-W
        t2 = t1 + (N-W)
        ahat = ((W-S)*t2 + (N-B-W)*t1) / (W2*t2 - t1**2)
        return ahat


class CellsLogLikelihood(object):
    """ A second attempt that attempts to do a better job of sampling the
        log likelihood.

        Talking through it -- if we don't sample the log likelihoods on
        a uniform grid, then for practicality if we use a value of alpha
        that is outside of the domain of one of the sub likelihoods, it
        is basically "minus infinity".  There are some contrived cases
        where you'd include something really significant with a different
        flux with a bunch of other significant things at a different flux.
        Then the formal solution would be in the middle, but it would be
        a terrible fit, so in terms of Bayesian blocks or anything else,
        you'd never want it.  So maybe it doesn't matter what its precise
        value is, and we can just return "minus infinity".

        This bears some checking, but for now, that's how it's coded up!

        What this means practically is that as we go along evaluating a
        fitness function, the bounds are always defined as the supremumm
        of the lower edges of the domains and the vice versa for the upper
        edges.  Still then have the problem of finding the maximum for
        irregular sampling.
    """
    def __init__(self,cells,profile_background=False,swap=False,
            reverse=False):
        """
        Parameters
        ----------
        cells : a list of Cell objects
        profile_background: use the profile likelihood instead of b=1
        swap : swap the source and the background
        reverse : change the time-order of the Cells; this is useful for
            checking that e.g. BB results are the same when looking
            forwards or backwards.  They ought to be!
        """

        # construct a sampled log likelihoods for each cell
        if reverse:
            cells = cells[::-1]
        self.clls = [CellLogLikelihood(x,swap=swap) for x in cells]
        npt = 200
        self._cod = np.empty([len(cells),npt])
        self._dom = np.empty([len(cells),npt])
        self._swap = swap
        self.cells = cells

        for icll,cll in enumerate(self.clls):
            self._dom[icll],self._cod[icll] = cll.get_logpdf(
                    npt=npt,dlogl=30,profile_background=profile_background)
        self.profile_background = profile_background

        self.fitness_vals = None

    def sanity_check(self):
        dx = self._dom[:,-1]-self._dom[:,0]
        bad_x = np.ravel(np.argwhere(np.abs(dx) < 1e-3))
        print('Indices with suspiciously narrow support ranges: ',bad_x)
        ymax = self._cod.max(axis=1)
        bad_y = np.ravel(np.argwhere(np.abs(ymax)>0.2))
        print('Indices where there is a substantial disagreement in the optimized value of the log likelihood and the codomain: ', bad_y)
    
    def fitness(self,i0,i1):
        """ Return the maximum likelihood estimator and value for the
        cell slice i0:i1, define such that ith element is:

        0 -- cells i0 to i1
        1 -- cells i1+1 to i1
        ...
        N -- i1

        """

        # set bounds on domain
        npt = 1000
        a0,a1 = self._dom[i1-1][0],self._dom[i1-1][-1]
        dom = np.linspace(a0,a1,npt)

        # initialize summed log likelihoods
        rvals = np.empty((2,i1-i0))
        cod = np.zeros(npt)

        for i in range(0,rvals.shape[1]):
            cod += np.interp(dom,self._dom[i1-1-i],self._cod[i1-1-i],
                    left=-np.inf,right=-np.inf)
            amax = np.argmax(cod)
            rvals[:,-(i+1)] = cod[amax],dom[amax]
            #rvals[-(i+1)] = cod[amax]
            #rvals[-(i+1)] = np.max(cod)

        return rvals*2

    def do_bb(self,prior=2):

        if self.fitness_vals is not None:
            return self._do_bb_cache(prior=prior)

        fitness = self.fitness

        ncell = len(self.cells)
        best = np.asarray([-np.inf]*(ncell+1))#np.zeros(nph+1)
        last = np.empty(ncell,dtype=int)
        fitness_vals = deque()
        last[0] = 0; best[0] = 0
        tmp = fitness(0,1)
        best[1] = tmp[0]-prior
        fitness_vals.append(tmp)
        for i in range(1,ncell):
            # form A(r) (Scargle VI)
            tmp = fitness(0,i+1)
            fitness_vals.append(tmp)
            a = tmp[0] - prior + best[:i+1]
            # identify last changepoint in new optimal partition
            rstar = np.argmax(a)
            best[i+1] = a[rstar]
            last[i] = rstar
        best_fitness = best[-1]

        cps = deque()
        last_index = last[-1]
        while last_index > 0:
            cps.append(last_index)
            last_index = last[last_index-1]
        indices = np.append(0,np.asarray(cps,dtype=int)[::-1])

        self.fitness_vals = fitness_vals

        # calculate overall variability TS
        var_dof = len(indices)-1
        var_ts = (best_fitness-tmp[0][0])+prior*var_dof

        return indices,best_fitness+len(indices)*prior,var_ts,var_dof,fitness_vals

    def _do_bb_cache(self,prior=2):

        fv = [x[0] for x in self.fitness_vals]

        ncell = len(self.cells)
        best = np.asarray([-np.inf]*(ncell+1))#np.zeros(nph+1)
        last = np.empty(ncell,dtype=int)
        last[0] = 0; best[0] = 0
        best[1] = fv[0]-prior
        for i in range(1,ncell):
            # form A(r) (Scargle VI)
            a = fv[i] - prior + best[:i+1]
            # identify last changepoint in new optimal partition
            rstar = np.argmax(a)
            best[i+1] = a[rstar]
            last[i] = rstar
        best_fitness = best[-1]

        cps = deque()
        last_index = last[-1]
        while last_index > 0:
            cps.append(last_index)
            last_index = last[last_index-1]
        indices = np.append(0,np.asarray(cps,dtype=int)[::-1])

        # calculate overall variability TS
        var_dof = len(indices)-1
        var_ts = (best_fitness-fv[-1][0])+prior*var_dof

        return indices,best_fitness+len(indices)*prior,var_ts,var_dof,self.fitness_vals

    def do_top_hat(self):
        """ Get a top-hat filtered representation of cells, essentially
            using the BB fitness function, but returning both the TS and
            the optimum value.
        """
        fitness_vals = self.do_bb()[-1]
        n = len(fitness_vals)
        rvals = np.empty((2,n,n))
        ix = np.arange(n)
        for iv,v in enumerate(fitness_vals):
            rvals[:,ix[:iv+1],ix[:iv+1]+n-(iv+1)] = v
            rvals[:,ix[:iv+1]+n-(iv+1),ix[:iv+1]] = np.nan
        # add an adjustment for dof to the TS map, and correct the flux
        rvals[0] += (np.arange(n)*2)[None,::-1]
        rvals[1] *= 0.5
        return rvals

    def get_flux(self,idx,conf=[0.05,0.95]):

        aguess = self._dom[idx][np.argmax(self._cod[idx])]
        if self.profile_background:
            bguess = self.clls[idx].get_beta_max(aguess)
        else:
            bguess = 1
        aopt = self.clls[idx].get_max(guess=aguess,beta=bguess,
                profile_background=self.profile_background)
        ## sanity check that aopt is close to the guess.  If not, it is
        ## wrong or else we extracted the domain/codomain incorrectly.
        #if abs(aopt[0]-aguess) > 0.1:
            #print aopt,aguess
            #print 'Warning! refinement of aguess failed.  Using internal version.'

        if self.profile_background:
            aopt = aopt[0]
        ts = self.clls[idx].get_ts(aopt=aopt,
                profile_background=self.profile_background)

        dom,cod = self._dom[idx],self._cod[idx].copy()
        np.exp(cod,out=cod)
        cod *= (1./simps(cod,x=dom))
        cdf = cumtrapz(cod,dom,initial=0)
        cdf *= 1./cdf[-1]
        indices = np.searchsorted(cdf,conf)
        # do a little linear interpolation step here
        ihi,ilo = indices,indices-1
        m = (cdf[ihi]-cdf[ilo])/(dom[ihi]-dom[ilo])
        xconf = dom[ilo] + (np.asarray(conf)-cdf[ilo])/m
        return aopt,ts,xconf

    def log_likelihood(self,alpha,slow_but_sure=False):
        """ Return the (positive) log likelihood for each cell.

        NB: the "fast" version of the log likelihood uses interpolators
            of the log likelihood which are set to 0 at the maximum value
            in each cell.  Therefore the returned values are relative to
            the maximum possible likelihood.  On the other hand, the
            "slow" version does not subtract off this maximum value.

        Parameters
        ----------
        alpha : an array equal in length to the number of cells giving the
            relative flux.  The sense is F(t) = alpha(t)*F_mean, viz. the
            nominal value of alpha is 1.
        slow_but_sure : if True, evaluate the log likelihood directly;
            otherwise, use linear interpolation of the previously-cached
            values
        """
        if len(alpha) != len(self.clls):
            raise ValueError('Must provide a value of alpha for all cells!')
        if not hasattr(self,'_interpolators'):
            # populate interpolators on first call
            self._interpolators = [interp1d(d,c) for d,c in zip(self._dom,self._cod)]
        if not slow_but_sure:
            rvals = 0
            try:
                for a,i in zip(alpha,self._interpolators):
                    rvals += i(a)
                return rvals
            except ValueError:
                pass
        if self.profile_background:
            return sum((cll.log_profile_likelihood(a) for cll,a in zip(self.clls,alpha)))
        else:
            return sum((cll.log_likelihood(a) for cll,a in zip(self.clls,alpha)))

    def get_raw_lightcurve(self,tsmin=4):
        """ Return a flux density light curve for the raw cells.

        Parameters
        ----------
        tsmin : minimum TS for returning uncertainty inteval or upper limit

        Returns
        -------
        time, terr, yval, yerrlo, yerrhi, TS
            time = MJD of the mid-time of the cell (or phase if phase)
            terr = width of the cell in days (or phase if phase)
            yval = fractional flux of the cell
            yerrlo = lower 1-sigma error (or upper limit)
            yerrhi = upper 1-sigma error (-1 if upper limit)
            TS = test statistic for cell
        """

        plot_phase = isinstance(self,PhaseCellsLogLikelihood)

        rvals = np.empty([len(self.clls),6])

        for icll,cll in enumerate(self.clls):
            if cll.S==0:
                rvals[icll] = np.nan
            tmid = cll.cell.get_tmid()
            if plot_phase:
                terr = (tmid-cll.cell.tstart)
            else:
                terr = (tmid-cll.cell.tstart)/86400
                tmid = met2mjd(tmid)
                #if plot_years:
                #    tmid = (tmid-54832)/365 + 2009 
                #    terr *= 1./365
            aopt,ts,xconf = self.get_flux(icll,conf=[0.159,0.841])
            ul = ts <= tsmin
            if ul:
                rvals[icll] = tmid,terr,xconf[1],0,-1,ts
            else:
                rvals[icll] = tmid,terr,aopt,aopt-xconf[0],xconf[1]-aopt,ts

        return rvals

    def get_bb_lightcurve(self,tsmin=4,bb_prior=8,get_indices=False):
        """ Perform a BB analysis and return the resulting light curve.

        Parameters
        ----------
        tsmin : minimum TS for returning uncertainty inteval or upper limit
        bb_prior : the exponential prior parameter for the BB algorithm
        get_indices : optionally return the indices of the raw cells for
            the BB partition boundaries

        Returns
        -------
        time, terr, yval, yerrlo, yerrhi, TS
            time = MJD of the mid-time of the cell (or phase if phase)
            terr = width of the cell in days (or phase if phase)
            yval = fractional flux of the cell
            yerrlo = lower 1-sigma error (or upper limit)
            yerrhi = upper 1-sigma error (-1 if upper limit)
            TS = test statistic for cell
        """

        plot_phase = isinstance(self,PhaseCellsLogLikelihood)

        bb_idx,bb_ts,var_ts,var_dof,fitness = self.do_bb(prior=bb_prior)
        #print('Variability significance: ',chi2.sf(var_ts,var_dof))
        bb_idx = np.append(bb_idx,len(self.cells))
        rvals_bb = np.empty([len(bb_idx)-1,6])
        for ibb,(start,stop) in enumerate(zip(bb_idx[:-1],bb_idx[1:])):
            cells = cell_from_cells(self.cells[start:stop])
            cll = CellLogLikelihood(cells,swap=self._swap)
            if cll.S==0:
                rvals_bb[ibb] = np.nan
                continue
            tmid = cll.cell.get_tmid()
            if plot_phase:
                terr = (tmid-cll.cell.tstart)
            else:
                terr = (tmid-cll.cell.tstart)/86400
                tmid = met2mjd(tmid)
                #if plot_years:
                    #tmid = (tmid-54832)/365 + 2009 
                    #terr *= 1./365
            aopt,ts,xconf = cll.get_flux(conf=[0.16,0.84],
                    profile_background=self.profile_background)
            if ts <= tsmin:
                rvals_bb[ibb] = tmid,terr,xconf[1],0,-1,ts
            else:
                rvals_bb[ibb] = tmid,terr,aopt,aopt-xconf[0],xconf[1]-aopt,ts

        if get_indices:
            return rvals_bb,bb_idx
        return rvals_bb


class PhaseCellsLogLikelihood(CellsLogLikelihood):
    """ Essentially the same as CellsLogLikelihood, but with a modified
        Bayesian Blocks method to handle the periodicity of the light
        curve."""

    def __init__(self,cells):

        super(PhaseCellsLogLikelihood,self).__init__(cells)

        # extend the domain/comain by one wrap on each side
        self._dom = np.concatenate([self._dom]*3)
        self._cod = np.concatenate([self._cod]*3)
        self._orig_cells = cells
        self.cells = [c.copy(phase_offset=-1) for c in cells]
        self.cells += cells
        self.cells += [c.copy(phase_offset=+1) for c in cells]
        self.profile_background = False
                

class Data(object):

    def __init__(self,ft1files,ft2files,ra,dec,weight_col,
            zenith_cut=100,theta_cut=0.4,minimum_exposure=3e4,use_phi=True,
            base_spectrum=None,max_radius=None,bary_ft1files=None,
            tstart=None,tstop=None,
            verbosity=1,correct_efficiency=True,correct_cosines=True,
            correct_psf=True,correct_aeff=False,
            use_psf_types=True,type_selector=None):
        """ 

        Generally, the times for everything should be topocentric.
        However, if there is a need to search for short period signals,
        then barycentric times should be used.  Exposure calculations must
        still be topocentric, so the easiest solution is simply to provide
        both files.

        Parameters
        ----------
        ft1files : the FT1 files
        ft2files : the FT2 files
        ra : R.A. of source (deg)
        dec : Decl. of source (deg)
        zenith_cut : maximum zenith angle for data/exposure calc. (deg)
        theta_cut : livetime cut on zenith angle (cosine(zenith))
        minimum_exposure : TODO
        use_phi : use phi-dependence in effective area calculation
        max_radius : maximum photon separation from source [deg]
        bary_ft1files : an optional mirrored set of FT1 files with
            photon timestamps in the barycentric reference frame
        tstart: cut time to apply to dataset; notionally MET, but will 
            attempt to convert from MJD if < 100,000.
        tstop : as tstart
        correct_efficiency: apply trigger efficiency (from livetime)
        correct_cosines: apply in-bin S/C attitude correction;  see below
        correct_psf : apply aperture completeness correction
        correct_aeff : apply very minor correction (~1%) estimated from
            Vela and Geminga
        use_psf_types : use PSF-types IRF for exposure calculation
        type_selector: an optional instance of PSFTypeSelector, e.g., which
            will be used to make an energy-dependence selection on event
            type

        The FT2 values are tabulated such that the S/C position is given
        at the START time.  Whereas the livetime etc. are accumulated over
        the full (~30s) interval.  The upshot is that the position is off
        by ~15s when evaluating the exposure.  Using correct_cosines will
        attempt to interpolate the S/C attitude between START and FINISH
        and will then use that for the exposure calculation.
        """
        self.ft1files = ft1files
        self.ft2files = ft2files
        self.max_radius = max_radius
        self.zenith_cut = zenith_cut
        self.theta_cut = theta_cut
        self.ra,self.dec = ra,dec
        self._verbosity = verbosity

        if tstart is not None:
            tstart = infer_met(tstart)
        if tstop is not None:
            tstop = infer_met(tstop)
        if type_selector is not None:
            if use_psf_types:
                assert(isinstance(type_selector,PSFTypeSelector))
            else:
                assert(isinstance(type_selector,FBTypeSelector))

        # Get DSS information from first FT1 file; the others will be
        # checked for consistency in _load_photons
        rcuts,ecuts,zmax,evtclass = events.parse_dss(
            fits.getheader(ft1files[0],1))
        emin,emax = ecuts # in MeV
        
        lt = py_exposure_p8.Livetime(ft2files,ft1files,
                tstart=tstart,tstop=tstop,verbose=verbosity)

        # Get the geometry and livetime information from the FT2 file
        lt_mask,pcosines,acosines,START,STOP,LIVETIME = lt.get_cosines(
                np.radians(ra),np.radians(dec),
                theta_cut=theta_cut,
                zenith_cut=np.cos(np.radians(zenith_cut)),
                get_phi=True,apply_correction=correct_cosines)

        phi = np.arccos(acosines) if use_phi else None

        if correct_efficiency:
            ltfrac = LIVETIME/(STOP-START)
            print(f'Median livetime fraction: {np.median(ltfrac):.2f}')
        else:
            ltfrac = None

        # TODO -- OK, here is where we want to put the PSFType selector
        # can actually make it general and take 0-3 or 0-1 depending on
        # the types being used
        aeff,texp_edom,texp = calc_weighted_aeff(
                pcosines,phi,
                base_spectrum=base_spectrum,ltfrac=ltfrac,
                correct_psf=correct_psf,use_psf_types=use_psf_types,
                emin=emin,emax=emax,type_selector=type_selector,
                verbosity=verbosity)

        self.total_exposure_edom = texp_edom
        self.total_exposure = texp
        exposure = aeff*LIVETIME

        if correct_aeff:
            correction = py_exposure_p8.aeff_corr(pcosines,acosines)
            exposure = exposure * correction

        # just get rid of any odd bins -- this cut corresponds to roughly
        # 30 seconds of exposure at the edge of the FoV; in practice it
        # doesn't remove too many photons
        self.minimum_exposure = minimum_exposure
        minexp_mask = exposure > minimum_exposure

        self.TSTART = START[minexp_mask]
        self.TSTOP = STOP[minexp_mask]
        self.LIVETIME = LIVETIME[minexp_mask]
        self.exposure = exposure[minexp_mask]

        # Load in the photons from the FT1 files
        data,datacols,self.timeref,_,_ = self._load_photons(
                ft1files,weight_col,self.TSTART[0],self.TSTOP[-1],
                max_radius=max_radius,zenith_cut=zenith_cut,
                type_selector=type_selector)
        ti = data[0]
        if self.timeref=='SOLARSYSTEM':
            print('WARNING!!!!  Barycentric data not accurately treated')
        
        # this is a problem if the data are barycentered
        event_idx = np.clip(np.searchsorted(self.TSTOP,ti),0,len(self.TSTOP)-1)
        event_mask = (ti >= self.TSTART[event_idx]) & (ti <= self.TSTOP[event_idx])
        # This should be redundant at this point
        event_mask &= lt.get_gti_mask(ti)
        self.event_mask = event_mask

        # TODO add a check for FT2 files that are shorter than observation
        data = [d[event_mask].copy() for d in data]
        self.ti = data[0]
        self.we = data[1]
        self.other_data = data[2:]
        self.datacols = datacols

        if bary_ft1files is not None:
            # This needs to be fixed before next use
            raise NotImplementedError()
            data,datacols,timeref,photon_idx,ecuts = self._load_photons(
                    bary_ft1files,weight_col,
                    None,None,max_radius=max_radius,no_filter=True,
                    zenith_cut=zenith_cut,type_selector=type_selector)
            self.bary_ti = (data[0][photon_idx][event_mask]).copy()
        else:
            self.bary_ti = None

        # do another sort into the non-zero times
        event_idx = self.event_idx = np.searchsorted(self.TSTOP,self.ti)

        # store the source and background exposure explicitly to enable
        # time-dependent modifications
        S = self.S = np.sum(self.we)
        B = self.B = len(self.we) - S
        E = self.E = np.sum(self.exposure)
        self.sexposure = None
        self.bexposure = None

    def _vprint(self,s,level):
        if self._verbosity >= level:
            print(s)

    def zero_weight(self,tstart,tstop):
        """ Zero out data between tstart and tstop.

        Parameters
        ----------
        tstart : beginning of zero interval (MET, but will infer from MJD)
        ttop : end of zero interval (MET, but will infer from MJD)
        """
        tstart = infer_met(tstart)
        tstop = infer_met(tstop)
        mask = ~((self.TSTART >= tstart) & (self.TSTOP <= tstop))
        event_mask = mask[np.searchsorted(self.TSTOP,self.ti)]

        # Adjust exposure
        self.exposure = self.exposure[mask]
        self.cexposure = np.cumsum(self.exposure)
        self.E = self.cexposure[-1]
        self.TSTART = self.TSTART[mask]
        self.TSTOP = self.TSTOP[mask]
        self.LIVETIME = self.LIVETIME[mask]

        # Delete events and adjust predicted counts
        self.ti = self.ti[event_mask]
        self.we = self.we[event_mask]
        self.S = np.sum(self.we)
        self.B = len(self.we) - self.S

    def __setstate__(self,state):
        if not 'sexposure' in state:
            state['sexposure'] = None
        if not 'bexposure' in state:
            state['bexposure'] = None
        self.__dict__.update(state)

    def _load_photons(self,ft1files,weight_col,tstart,tstop,
            max_radius=None,time_col='time',no_filter=False,
            zenith_cut=100,type_selector=None):
        """ Load events from the FT1 files.

        Parameters
        ----------
        ft1files : a list of FT1 files
        weight_col : the FT1 column giving the source photon probability
        tstart : (MET) cut photons before this time (None OK)
        tstop :  (MET) cut photons after  this time (None OK)
        max_radius : (deg)
        time_col : name of the FT1 column giving the event time
        no_filter : do not sort or perform the time cut

        Returns
        -------
        data : a list of FT1 columns, the entries of which are the rows
            which survive the specified cuts; they are TIME, "WEIGHT", and
            ENERGY
        timeref : the TIMEREF entry of the FT1 files (same for all)
        idx : an array which can be used to sort other FT1 columns
        dss : the DSS selections (same for all)
        """

        cols = [time_col,weight_col,'energy','event_type']
        deques = [deque() for col in cols]

        # check DSS cuts against the ones in the first file
        with fits.open(ft1files[0]) as f:
            hdr = f[1]._header
            rcuts0,ecuts0,zmax0,evtclass0 = dss0 = events.parse_dss(hdr)
            timeref0 = hdr['timeref']

        def _check_metadata(hdr,tol=1e-4):
            """ Check for consistency in the DSS keywords, zenith cuts,
                time systems, and for the presence of the specified photon
                weights column.
            """
            rcuts,ecuts,zmax,evtclass = dss = events.parse_dss(hdr)
            if not (ecuts0[0]==ecuts[0] and ecuts0[1]==ecuts[1]):
                raise ValueError('Data energy cuts were not consistent.')
            ra0,de0,rad0 = rcuts0
            ra1,de1,rad1 = rcuts
            if not (abs(ra0-ra1) < tol):
                raise ValueError(f'Data RA values are not consistent: {ra0:.5f} != {ra1:.5f}')
            if not (abs(de0-de1) < tol):
                raise ValueError(f'Data Dec values are not consistent: {de0:.5f} != {de1:.5f}')
            if not (abs(rad0-rad1) < tol):
                raise ValueError(f'Data aperture radius values are not consistent: {rad0:.5f} != {rad1:.5f}')
            if (evtclass is not None):
                if not (evtclass0[0]==evtclass[0] and evtclass0[1]==evtclass[1]):
                    raise ValueError('Event class selection was not consistent.')
            # raise an error if we've already cut more than requested; if
            # it's the other way around, remove the larger zenith angle
            # cuts later
            if zmax < zenith_cut:
                raise ValueError(f'zenith_cut argument {zenith_cut} < {zmax} (data)')
            if hdr['timeref'] != timeref0:
                raise Exception('Different time systems!')
            colnames = [x.name.lower() for x in f[1].columns]
            if weight_col.lower() not in colnames:
                raise ValueError(f'FT1 file {ft1} did not have weights column {weight_col}!')
            return dss

        for ift1,ft1 in enumerate(ft1files):

            f = fits.open(ft1)
            hdu = f['events']
            rcuts,ecuts,zmax,evtclass = _check_metadata(hdu._header)
            nevt = hdu._header['naxis2']
            mask = np.full(nevt,True,dtype=bool)

            if max_radius is not None:
                mask &= events.radius_cut(hdu,self.ra,self.dec,max_radius)
                self._vprint(f'{mask.sum()}/{nevt} pass radius cut',2)

            # requesting a zenith cut more stringent than was applied
            if zenith_cut < zmax:
                zmask = hdu.data['zenith_angle'] <= zenith_cut
                self._vprint(f'{zmask.sum()}/{nevt} pass zenith cut',2)
                mask &= zmask

            # merge the columns into the cumulative deques
            for c,d in zip(cols,deques):
                if c == 'event_type':
                    # load in as an integer array instead of 2d boolean
                    dat = events.load_bitfield(ft1,c)[mask]
                else:
                    dat = hdu.data[c][mask]
                d.append(dat)

            f.close()

        data = [np.concatenate(d) for d in deques]

        if type_selector is not None:
            ti,we,en,et = data
            if isinstance(type_selector,PSFTypeSelector):
                et = events.event_type_to_psf_type(et)
            else:
                et = events.event_type_to_fb_type(et)
            mask = type_selector.accept(en,et)
            for q in range(4):
                m = et == q
                print(f'{m.sum()} type={q} evts, keeping {mask[m].sum()}.')
                print(f'{we[m].sum():.1f} type={q} wts, keeping {(we*mask)[m].sum():.1f}.')
            print(f'accepting {mask.sum()}/{len(mask)} for ET selection.')
            data = [d[mask] for d in data]

        if no_filter:
            return data,timeref0

        # sort everything on time (in case files unordered)
        ti = data[0]
        a = np.argsort(ti)
        if (tstart is not None) and (tstop is not None):
            self._vprint('applying time cut',2)
            tstart = tstart or 0
            tstop = tstop or 999999999
            idx = a[(ti >= tstart) & (ti <= tstop)]
        else:
            idx = a
        data = [d[idx] for d in data]
        return data,cols,timeref0,idx,dss0

    def _bary2topo(self,bary_times):
        if not hasattr(self,'_b2t'):
            self._b2t = self.get_bary2topo_interpolator()
            self._vprint('Warning!  This formulation does not account for travel time around the earth; all conversions done for geocenter.',2)
        return self._b2t(bary_times)

    def _topo2bary(self,topo_times):
        if not hasattr(self,'_t2b'):
            self._t2b = self.get_topo2bary_interpolator()
            self._vprint('Warning!  This formulation does not account for travel time around the earth; all conversions done for geocenter.',2)
        return self._t2b(topo_times)

    def get_bary2topo_interpolator(self):
        return bary.totopo_interpolator(self.TSTART[0],self.TSTOP[-1],
                self.ra,self.dec)

    def get_topo2bary_interpolator(self):
        return bary.tobary_interpolator(self.TSTART[0],self.TSTOP[-1],
                self.ra,self.dec)

    def get_exposure(self,times,deadtime_too=False,cumulative=True):
        """ Return the three exposures at the given times. These
        are the original source exposure, the (possibly scaled) source
        exposure, and the (possibly scaled) background exposure.

        Returns
        -------
        cexp : the exposure
        csexp : the exposure scaled by S/E (total counts / total exposure)
        cbexp : the exposure scaled by B/E (total bkg / total exposure)
        cdead : the deadtime (possibly None)
        """

        TSTART,TSTOP = self.TSTART,self.TSTOP
        idx = np.searchsorted(TSTOP,times)
        if np.any(idx>=len(TSTOP)):
            m = idx >= len(TSTOP)
            print(m.sum())
            print(times[m]-TSTOP[-1])
        # if time falls between exposures, the idx will be of the following
        # livetime entry, but that's OK, because we'll just mask on the
        # negative fraction; ditto for something that starts before
        frac = (times -TSTART[idx])/(TSTOP[idx]-TSTART[idx])
        np.clip(frac,0,1,out=frac)
        # take the complementary fraction
        frac = 1-frac

        if self.sexposure is None:
            # populate this on the first call
            self.sexposure = (self.S/self.E) * self.exposure

        if self.bexposure is None:
            self.bexposure = (self.B/self.E) * self.exposure

        cexp = np.cumsum(self.exposure)[idx] - self.exposure[idx]*frac
        csexp = np.cumsum(self.sexposure)[idx] - self.sexposure[idx]*frac
        cbexp = np.cumsum(self.bexposure)[idx] - self.bexposure[idx]*frac

        if deadtime_too:
            deadtime = (TSTOP-TSTART)-self.LIVETIME
            cdead = np.cumsum(deadtime)[idx]-deadtime[idx]*frac
        else:
            cdead = None

        if not cumulative:
            cexp = cexp[1:]-cexp[:-1]
            csexp = csexp[1:]-csexp[:-1]
            cbexp = cbexp[1:]-cbexp[:-1]
            if cdead is not None:
                cdead = cdead[1:]-cdead[:-1]

        return cexp,csexp,cbexp,cdead

    def get_deadtime(self,times):
        """ Return the cumulative deadtime at the provided times."""
        return self.get_exposure(time,deadtime_too=True)[-1]

    def get_contiguous_exposures(self,tstart=None,tstop=None,
            max_interval=10):
        """ Return those intervals where the exposure is uninterrupted.
        This will typically be an orbit, or possibly two portions of an
        orbit if there is an SAA passage.

        max_interval -- maximum acceptable break in exposure for a
            contiguous interval [10s]
        """
        return py_exposure_p8.get_contiguous_exposures(
                self.TSTART,self.TSTOP,tstart=tstart,tstop=tstop,
                max_interval=max_interval)

    def get_photon_cells(self,tstart,tstop,max_interval=30):
        """ Return starts, stops, and exposures for photon-based cells.

        max_interval -- maximum time interval between good exposure bins
            before breaking the photon cell [default 30 s]

        This is essentially the Voronoi tessellation concept, where change
        points occur between photons, with "between" being the mid-point in
        exposure rather than time.

        There is some ambiguity about what the right thing to do here is.
        Consider an orbit interrupted by an SAA passage, and photons
        arriving on either side of it.  For a rapidly varying source, the
        cells should clearly end at/begin after the SAA, regardless of the
        exposure.  For a faint or steady source, it probably doesn't matter.

        Though rare, there is also a possibility of multiple breaks.  For
        concreteness, and to take care of almost all cases, I am going to
        break a photon cell if there is a long break [max_interval], and
        all of the exposure before the first break goes with the preceeding
        photon and all of the subsequent exposure, regardless of additional
        breaks, goes with the second photons.
        """

        if tstart < self.TSTART[0]:
            raise ValueError('Start time precedes start of exposure!')
        if tstop > self.TSTOP[-1]:
            raise ValueError('Stop time exceeds stop of exposure!')

        # find the events within that interval
        event_idx0,event_idx1 = np.searchsorted(self.ti,[tstart,tstop])
        ti = self.ti[event_idx0:event_idx1]
        print('have %d photons'%(len(ti)))

        # snap tstart/stop to edges of exposure
        idx_start,idx_stop = np.searchsorted(self.TSTOP,[tstart,tstop])
        tstart = max(tstart,self.TSTART[idx_start])
        if tstop < self.TSTART[idx_stop]:
            idx_stop -= 1
        tstop = min(tstop,self.TSTOP[idx_stop])

        # calculate exposure at start/stop + photon times
        times = np.empty(len(ti)+2)
        times[0] = tstart
        times[-1] = tstop
        times[1:-1] = ti
        exposure = self.get_exposure(times)[0]

        # now, calculate the exposure at the mid-points
        mp_exposure = 0.5*(exposure[1:]+exposure[:-1])

        # and invert it to get the mid-point times
        c = self.cexposure
        t0 = self.TSTART
        t1 = self.TSTOP
        idx = np.searchsorted(c,mp_exposure)
        frac = 1-(c[idx]-mp_exposure)/self.exposure[idx]
        mp_times = self.TSTART[idx]+frac*(self.TSTOP[idx]-self.TSTART[idx])

        # discard the initial and last mid-point times because they should
        # be snapped to the boundary, and this is our initial guess at the
        # cell boundaries
        cell_starts = mp_times[:-1].copy()
        cell_starts[0] = tstart
        cell_stops = mp_times[1:].copy()
        cell_stops[-1] = tstop

        # now, all we need to do is make sure that there are no breaks
        # between the photon time stamps and the initial cell_stops
        t0s = self.TSTART[idx_start:idx_stop+1]
        t1s = self.TSTOP[idx_start:idx_stop+1]
        break_mask = (t0s[1:]-t1s[:-1])>max_interval
        break_starts = t1s[:-1][break_mask]
        break_stops = t0s[1:][break_mask]
        # indices of cells with breaks
        idx = np.searchsorted(cell_stops,break_stops)

        # if the photon follows the break, move the previous cell stop 
        # time up to break and move the current start time to end of break
        m_follow = ((ti[idx]-break_stops) > 0) | (idx == len(ti) - 1)
        cell_stops[idx[m_follow]-1] = break_starts[m_follow]
        cell_starts[idx[m_follow]] = break_stops[m_follow]
        # if the photon precedes the break, move the current cell stop 
        # to the break start and the following cell start time to the end
        m_precede = ~m_follow
        cell_stops[idx[m_precede]] = break_starts[m_precede]
        cell_starts[idx[m_precede]+1] = break_stops[m_precede]

        # now just need to compute exposure; this can be done just by
        # tweaking the times above, but let's just make it easy for now
        exposure = self.get_exposure(cell_stops)[0] - self.get_exposure(cell_starts)[0]

        # there should now be a 1-1 mapping between tstart, tstop,
        # photon times, photon weights, and exposures
        return list(map(Cell,cell_starts,cell_stops,exposure*(self.S/self.E),ti,self.we[event_idx0:event_idx1],[self.S/self.B]*len(cell_starts)))

    def get_cells(self,tstart=None,tstop=None,tcell=None,
            trim_zero_exposure=True,
            time_series_only=False,use_barycenter=True,
            randomize=False,seed=None,
            src_scaler=None,bkg_scaler=None,src_injector=None,
            minimum_exposure=3e4,minimum_fractional_exposure=0,
            quiet=False):
        """ Return the starts, stops, exposures, and photon data between
            tstart and tstop.  If tcell is specified, bin data into cells
            of length tcell (s).  Otherwise, return photon-based cells.

            Parameters
            ----------
            trim_zero_exposure : remove Cells that have 0 exposure.  good
                for things like Bayesian Block analyses, VERY BAD for
                frequency-domain analyses!

            time_series_only : don't return Cells, but a list of time
                series: starts, stops, exposure, cts, sum(weights), 
                sum(weights^2)

            use_barycenter : interpret tcell in barycenter frame, so
                generate a set of nonuniform edges in the topocenter and
                use this for binning/exposure calculation; NB that in
                general one would *not* use barycentered event times
                in this case.

                **NB** This calculation will be off by up to 40ms 
                because it assumes that the S/C is at the geocenter.

            randomize : shuffle times and weights so that they follow
                the exposure but lose all time ordering; useful for
                exploring null cases

            seed : [None] random seed to use (see np.random.default_rng)

            src_scaler : a function which can takes METs (topo) of the
                start and stop of the exposure/other interval, giving the
                average source scale over that interval.  This will be 
                applied both to the exposure and to the weights:

                w' = a*w / (a*w + b*(1-w)),

                where a,b are the source and background scales for the
                weights.

                The intent for src_scaler and bkg_scaler is to "redo"
                the weights computation as if we had known about the 
                variability originally.  This is exact for the source
                (with the restriction to a constant spectrum) but only
                approximate for the background, since the individual source
                contributions have been summed up.

                One example application is Cygnus X-3, where there
                is overall "slow" source variation and a fast (orbital)
                periodicity.  Scaling so that the weights account for the
                slow variation (enhancing the times when it is on) improves
                the periodicity sensitivity.
                
            bkg_scaler : as src_scaler, but will rescale the "background"
                sources, viz. those represented by the complement of the
                weights

                Generally, one would use this to suppress variations from
                background sources.

            src_injector : as src_scaler/bkg_scaler, but the intention is
                to inject a signal without changing the underlying weights
                or exposure.  If provided, the source exposure will be
                temporarily adjusted and the weights will be restributed
                according to the updated source/bkg distributions.

                *** NB -- currently, unlike src/bkg_scaler, the argument
                to src_injector is expected to be barycentric.
                Generally this is kindof a mess...
                ***

            minimum_fractional_exposure -- reject cells whose exposure
                is less than this fraction of the mean exposure of all of
                the cells.  Only valid when tcell is specified.  Use with
                care if using short (e.g. <1d) tcell, as the intrinsic
                exposure variation becomes large.

        """
        ft1_is_bary = self.timeref == 'SOLARSYSTEM'

        if tstart is None:
            tstart = self.TSTART[0]
        tstart = infer_met(tstart)
        if tstart < self.TSTART[0]:
            print('Warning: Start time precedes start of exposure.')
            print('Will clip to MET=%.2f.'%(self.TSTART[0]))
            tstart = self.TSTART[0]

        if tstop is None:
            tstop = self.TSTOP[-1]
        tstop = infer_met(tstop)
        if tstop > self.TSTOP[-1]:
            print('Warning: Stop time follows end of exposure.')
            print('Will clip to MET=%.2f.'%(self.TSTOP[-1]))
            tstop = self.TSTOP[-1]

        if use_barycenter:
            tstart,tstop = self._topo2bary([tstart,tstop])

        if tcell is None:
            return self.get_photon_cells(tstart,tstop)

        ncell = int((tstop-tstart)/tcell)
        if ncell == 0:
            tcell = tstop-tstart
            ncell = 1

        edges = tstart + np.arange(ncell+1)*tcell
        assert(edges[-1] <= tstop)
        if use_barycenter:
            topo_edges = self._bary2topo(edges)
            assert(abs(topo_edges[0]-self.TSTART[0])<1e-6)
            topo_edges[0] = self.TSTART[0] # fix tiny numerical error
            assert(topo_edges[-1] <= self.TSTOP[-1])
            bary_edges = edges
        else:
            topo_edges = bary_edges = edges
        self._topo_edges = topo_edges.copy()

        # apply scales if provided
        ov = self.apply_scalers(
                src_scaler=src_scaler,bkg_scaler=bkg_scaler)

        # always use topocentric times to manage the exposure calculation
        exp,sexp,bexp,dead = self.get_exposure(
                topo_edges,deadtime_too=True,cumulative=False)

        if ft1_is_bary:
            starts,stops = bary_edges[:-1],bary_edges[1:]
        else:
            starts,stops = topo_edges[:-1],topo_edges[1:]

        # trim off any events that come outside of first/last cell
        istart,istop = np.searchsorted(self.ti,[starts[0],stops[-1]])
        times = self.ti[istart:istop]
        weights = self.we[istart:istop]

        if minimum_fractional_exposure > 0:
            if randomize:
                print("Warning!  Using minimum_fractional_exposure>0 with randomize is not tested, probably not supported.")
            frac_exp = exp/exp[exp>0].mean()
            m = frac_exp < minimum_fractional_exposure
            exp[m] = 0
            sexp[m] = 0
            bexp[m] = 0

        if trim_zero_exposure:
            exposure_mask = exp > 0
            # need to remove events that will lie outside of cells after
            # exposure cut
            event_mask = exposure_mask[np.searchsorted(stops,times)]
            times = times[event_mask]
            weights = weights[event_mask]
            starts = starts[exposure_mask]
            stops = stops[exposure_mask]
            exp = exp[exposure_mask]
            sexp = sexp[exposure_mask]
            bexp = bexp[exposure_mask]
            dead = dead[exposure_mask]
        else:
            exposure_mask = slice(0,len(starts))

        if src_injector is not None:
            randomize = True

        if randomize:
            # Redistribute the weights according to the (possibly rescaled)
            # source and background exposures

            if src_injector is not None:
                csexp = np.cumsum(
                        src_injector(bary_edges[:-1],bary_edges[1:])*sexp)
            else:
                csexp = np.cumsum(sexp)
            csexp *= 1./csexp[-1]
            cbexp = np.cumsum(bexp)
            cbexp *= 1./cbexp[-1]

            rng = np.random.default_rng(seed)
            mask = rng.random(len(times)) >= weights # bkg events
            rv = rng.random(len(times)) # random insertion points

            # Insert the weights randomly following the distributions
            indices = np.empty(len(times),dtype=int)
            indices[mask] = np.searchsorted(cbexp,rv[mask])
            mask = ~mask
            indices[mask] = np.searchsorted(csexp,rv[mask])

            a = np.argsort(indices)
            times = times[a]
            weights = weights[a]
            event_idx = indices[a]
            assert(np.all(exp[event_idx]>0))

        else:
            event_idx = np.searchsorted(stops,times)

        nweights = np.bincount(event_idx,minlength=len(starts))

        # now that exposure and event selection are done in topocentric,
        # if we're using barycentering, replace the bin edges with the
        # correct (uniform) barycentric times;
        if use_barycenter and (not ft1_is_bary):
            # replace starts/stops with barycenter times
            starts = bary_edges[:-1][exposure_mask]
            stops = bary_edges[1:][exposure_mask]

        # restore the weights/exposure if we had been using src/bkg_scaler
        self.revert_scalers(ov)

        # AHAH -- this is broken when the weights have been re-shuffled
        # Needs to obey event_idx, so probably back to the slow mode
        if time_series_only:
            # Get the indices of the cells in the weights
            idx = np.append(0,np.cumsum(nweights))
            # Use high precision on the cumulative sum
            W1 = np.append(0,np.cumsum(weights.astype(np.float128)))
            W2 = np.append(0,np.cumsum(weights.astype(np.float128)**2))
            weights_vec = (W1[idx[1:]]-W1[idx[:-1]]).astype(float)
            weights2_vec = (W2[idx[1:]]-W2[idx[:-1]]).astype(float)
            if use_barycenter:
                return CellTimeSeries(
                    starts,stops,exp,sexp,bexp,
                    nweights,weights_vec,weights2_vec,dead,
                    alt_starts=topo_edges[:-1][exposure_mask],
                    alt_stops=topo_edges[1:][exposure_mask],
                    timesys='barycenter',minimum_exposure=0)
            else:
                return CellTimeSeries(
                    starts,stops,exp,sexp,bexp,
                    nweights,weights_vec,weights2_vec,dead,
                    timesys='topocentric',minimum_exposure=0)

        cells = deque()
        idx = 0
        for i in range(len(starts)):
            t = times[idx:idx+nweights[i]]
            w = weights[idx:idx+nweights[i]]
            idx += nweights[i]
            SonB = sexp[i]/bexp[i]
            c = Cell(starts[i],stops[i],sexp[i],t.copy(),w.copy(),SonB)
            cells.append(c)
        return list(cells)

    def get_cells_from_time_intervals(self,tstarts,tstops,
            randomize=False,seed=None):
        """ Given a specific set of start and stop times, makes Cells.

        Ideally this would be merged with the more general method.
        """

        exp = self.get_exposure(tstops)[1]-self.get_exposure(tstarts)[1]
        #exp *= self.S/self.E
        start_idx = np.searchsorted(self.ti,tstarts)
        stop_idx = np.searchsorted(self.ti,tstops)
        if randomize:

            rng = np.random.default_rng(seed)

            # first, just get all of the relevant weights and times
            nphot = np.sum(stop_idx)-np.sum(start_idx)
            ncell = len(tstarts)
            we = np.empty(nphot)
            ti = np.empty(nphot)
            counter = 0
            for i in range(ncell):
                i0,i1 = start_idx[i],stop_idx[i]
                myn = i1-i0
                we[counter:counter+myn] = self.we[i0:i1]
                ti[counter:counter+myn] = self.ti[i0:i1]
                counter += myn

            # now, use the exposure to uniformly distribute the photons
            # according to it
            cexp = np.cumsum(exp)
            cexp /= cexp[-1]
            new_indices = np.searchsorted(cexp,rng.random(nphot))
            nweights = np.bincount(new_indices,minlength=ncell)

            # randomly permute the weights, and then draw random times
            # for each cell for the give number of photons in it
            a = np.argsort(new_indices)
            new_indices = new_indices[a]
            we = we[a]
            dts = (tstops-tstarts)[new_indices]
            ti = tstarts[new_indices] + rng.random(nphot)*dts

            cells = deque()
            idx = 0
            for i,inphot in enumerate(nweights):
                cells.append(Cell(tstarts[i],tstops[i],exp[i],
                    ti[idx:idx+inphot],we[idx:idx+inphot],self.S/self.B))
                idx += inphot
                
            return list(cells)

                
        cells = deque()
        for i in range(len(tstarts)):
            i0,i1 = start_idx[i],stop_idx[i]
            cells.append(Cell(tstarts[i],tstops[i],exp[i],
                self.ti[i0:i1].copy(),self.we[i0:i1].copy(),self.S/self.B))
            # TMP
            #m = (self.ti>=tstarts[i]) & (self.ti < tstops[i])
            #assert(m.sum() == len(cells[-1].ti))
            # end TMP
        return list(cells)

    def get_contiguous_exposure_cells(self,tstart=None,tstop=None,
            randomize=False,seed=None):
        """ Return Cells for all contiguous exposure cells between
        tstart and tstop.
        """
        starts,stops = self.get_contiguous_exposures(
                tstart=tstart,tstop=tstop)
        return self.get_cells_from_time_intervals(starts,stops,
                randomize=randomize,seed=seed)

    def apply_scalers(self,src_scaler=None,bkg_scaler=None):
        """ Apply permanent changes to the underlying exposure and weights.
        """
        if (src_scaler is None) and (bkg_scaler is None):
            return

        if self.sexposure is None:
            self.sexposure = self.exposure * (self.S / self.E)
        if self.bexposure is None:
            self.bexposure = self.exposure * (self.B / self.E)

        original_values = (self.we,
                self.sexposure.copy(),self.bexposure.copy())

        sscl = bscl = 1.
        if src_scaler is not None:
            self.sexposure *= src_scaler(self.TSTART,self.TSTOP)
            sscl = src_scaler(self.ti)
        if bkg_scaler is not None:
            self.bexposure *= bkg_scaler(self.TSTART,self.TSTOP)
            bscl = bkg_scaler(self.ti)

        self.we = sscl*self.we / (sscl*self.we + bscl*(1-self.we))

        return original_values

    def revert_scalers(self,original_values):
        """ Just a convenience method for undoing apply_scalers.
        """
        if original_values is None:
            return
        we,sexp,bexp = original_values
        self.we = we
        self.sexposure = sexp
        self.bexposure = bexp


class PhaseData(Data):
    """ Use phase instead of time, and ignore exposure.
    """

    def __init__(self,ft1files,weight_col,max_radius=None,
            pulse_phase_col='PULSE_PHASE',phase_shift=None,
            ra=None,dec=None,verbosity=1):
        """ The FT1 files
            ra, dec of source (deg)
            max_radius -- maximum photon separation from source [deg]
        """
        print('WARNING! This class is probably out of date.')
        self.ft1files = ft1files
        self.ra = ra
        self.dec = dec
        if (max_radius is not None) and ((ra or dec) is None):
            raise ValueError('Must specify ra and dec for radius cut!')
        self._verbosity = verbosity

        data = self._load_photons(ft1files,weight_col,None,None,
                max_radius=max_radius,time_col=pulse_phase_col)[0]

        if phase_shift is not None:
            ph = data[0]
            ph += phase_shift
            np.mod(ph,1,out=ph)
            a = np.argsort(ph)
            data = [d[a].copy() for d in data]
        self.ti = data[0]
        self.we = data[1]
        self.other_data = data[2:]

        self.E = 1
        self.S = np.sum(self.we)
        self.B = len(self.we)-self.S



    def get_cells(self,ncell=100,get_converse=False,randomize=False,seed=None):
            
        edges = np.linspace(0,1,ncell+1)
        starts,stops = edges[:-1],edges[1:] 

        times = self.ti
        weights = self.we
        if get_converse:
            weights = 1-weights
        if randomize:
            rng = np.random.default_rng(seed)
            times = rng.random(len(times))
            a = np.argsort(times)
            times = times[a]
            weights = weights[a]
        event_idx = np.searchsorted(stops,times)
        nweights = np.bincount(event_idx,minlength=len(starts))

        cells = deque()
        idx = 0
        exp = self.S/ncell
        SonB = self.S/self.B

        for i in range(len(starts)):
            t = times[idx:idx+nweights[i]]
            w = weights[idx:idx+nweights[i]]
            idx += nweights[i]
            c = PhaseCell(starts[i],stops[i],exp,t.copy(),w.copy(),SonB)
            cells.append(c)
        return list(cells)

class CellsLogLikelihoodOld(object):
    """ An optimized class to handle models that specify intensities for
    more than one Cell.

    The basic idea is to have a class that distributes the alpha value
    correctly so that it can be done is just a few monolithic array
    operations.
    """
    def __init__(self,cells):
        self.cells = cells
        self.nph = np.sum([len(c.we) for c in cells])
        self.exp = np.asarray([c.exp for c in cells])
        self.weights = np.empty(self.nph,dtype=np.float64)
        self.alpha_indices = np.full(self.nph,-1,dtype=int)
        ctr = 0
        for i,ic in enumerate(cells):
            nph = len(ic.we)
            s = slice(ctr,ctr+nph)
            self.weights[s] = ic.we
            self.alpha_indices[s] = i
            ctr += nph
        assert(not(np.any(self.alpha_indices==-1)))

        self._tmp1 = np.empty(self.nph,dtype=np.float64)
        self.bweights = 1-self.weights
        self.tmids = np.asarray([c.get_tmid() for c in self.cells])

    def log_likelihood(self,alpha):
        """ Alpha is a vector equal to the number of cells."""
        # distribute alpha to the correct shape
        t1 = self._tmp1
        t1[:] = alpha[self.alpha_indices]
        np.multiply(t1,self.weights,out=t1)
        np.add(t1,self.bweights,out=t1)
        np.log(t1,out=t1)
        return np.sum(t1)-np.sum(self.exp*alpha)


def fit_harmonics(cll,freq):
    """ Find the best-fitting sin/cos for the given frequency."""

    # to do this quickly, assume a will be small, which allows the denom.
    # in the first derivative of the log likelihood to be expanded and
    # solved analytically.  This should work well except for detections,
    # which can be revisited iteratively.  (E.g. a posthoc sanity check.)

    # (1) compute the phase of all of the cells
    #tmids = np.asarray([c.get_tmid() for c in cll.cells])
    tmids = cll.tmids
    # cache the phasing operation when doing actual power spectrum
    ph = (2*np.pi*freq)*(tmids-tmids.mean())
    cph = np.cos(ph)
    sph = np.sin(ph)

    # take moments -- if want, can use buffer in cll object, but will prob.
    # be better to integrate this directly into class
    tmp = cph[cll.alpha_indices]
    tmp *= cll.weights
    cw1 = np.sum(tmp)
    tmp *= tmp
    cw2 = np.sum(tmp)
    ce = np.sum(cll.exp*cph)
    # if doing quadratic, here is solution
    # tmp *= tmp
    # cw3 = np.sum(tmp)
    # a = cw3; b = -cw2; c = cw1-ce
    # camp = (0.5/a)*(-b-(b**2-4*a*c)**0.5)
    camp = (cw1-ce)/cw2

    # now do same thing for sine
    tmp[:] = sph[cll.alpha_indices]
    tmp *= cll.weights
    sw1 = np.sum(tmp)
    tmp *= tmp
    sw2 = np.sum(tmp)
    se = np.sum(cll.exp*sph)
    samp = (sw1-se)/sw2

    # if we want, we can estimate them jointly, but in practice they are
    # so darned close it doesn't matter; maybe not the case if close
    # to a real signal
    #tmp[:] = sph[cll.alpha_indices]*cph[cll.alpha_indices]*cll.weights**2
    #csw1 = np.sum(tmp)
    #camp_j = (cw1-ce)/(cw2-(csw1**2/sw2))
    #samp_j = (sw1-se)/sw2-csw1/sw2*camp

    # evaluate log likelihood change under the hypothesis;  NB using the
    # same approximation of small amplitude, this is actually quite a
    # nice expression!  In order for it to work out, need to keep second-
    # order term in logarithm expansion, but using the ML estimators, they
    # contribute -1/2 of the leading order term!  The only thing left is
    # a small cross-term, which I leave in the comments below.
    # TS = 2*dlog
    ts = (cw1-ce)**2/cw2 + (sw1-se)**2/sw2
    # the correction term from the second order neglected in the above is
    #ts_q = (cw1-ce)**2/cw2 + (sw1-se)**2/sw2 -2*camp*samp*csw1
    # additionally, if want to use the joint estimators, it's probably
    # easiest to insert them directly
    #ts_j = (cw1-ce)*camp_j + (sw1-se)*samp_j -2*camp_j*samp_j*csw1

    return camp,samp,ts

def compute_ls(cll,freqs,unweighted=False):
    """ Return Lomb-Scargle periodogram at the specified frequencies."""
    # form fluxes
    if unweighted:
        flux = np.asarray([len(c.we) for c in cll.cells],dtype=float)/cll.exp**0.5
    else:
        #flux = np.asarray([np.sum(c.we) for c in cll.cells])/cll.exp**0.5
        #flux = np.asarray([np.sum(c.we) for c in cll.cells])/cll.exp
        flux = np.asarray([np.sum(c.we) for c in cll.cells])-cll.exp
    fscale = flux.mean()
    flux -= flux.mean()
    flux /= flux.std()

    ph_basis = (cll.tmids-cll.tmids[0])

    cph = np.empty(len(ph_basis))
    sph = np.empty(len(ph_basis))
    tph = np.empty(len(ph_basis),dtype=int)
    
    rvals = np.empty(len(freqs))

    # evaluate cosine and sine on a small range
    cos_cod = np.cos(np.linspace(0,2*np.pi,1001)[:-1])
    sin_cod = np.sin(np.linspace(0,2*np.pi,1001)[:-1])

    for i in range(len(rvals)):

        tph[:] = np.multiply(ph_basis,freqs[i],out=sph)
        sph -= tph
        np.multiply(sph,1000,out=tph,casting='unsafe')
        cph[:] = cos_cod[tph]
        sph[:] = sin_cod[tph]

        cph *= flux
        sph *= flux

        rvals[i] = np.sum(cph)**2 + np.sum(sph)**2

    # this should handle Leahy normalization
    scale = 0.5*np.sum(flux**2)

    return rvals/scale

def power_spectrum_dft(cll,freqs,unweighted=False):
    """ Do a computation of the power spectrum using first-order likelihood
        expansion on potentially un-even cell sizes.  This also allows 
        arbitrary frequencies (DFT).  But does make the approx. that the 
        cosine/sine is constant within a cell like the FFT method.
    """

    ph_basis = (cll.tmids-cll.tmids[0])
    dt = cll.tmids[1]-cll.tmids[0]

    tmp = np.empty(cll.nph)
    cph = np.empty(len(ph_basis))
    sph = np.empty(len(ph_basis))
    tph = np.empty(len(ph_basis),dtype=int)
    
    if unweighted:
        weights = np.ones_like(cll.weights)
        # need to (re)correct teh exposure scale
        exp = cll.exp*(float(cll.nph)/np.sum(cll.weights))
    else:
        weights = cll.weights
        exp = cll.exp

    alpha_indices = cll.alpha_indices
    
    rvals = np.empty(len(freqs))

    # evaluate cosine and sine on a small range
    cos_cod = np.cos(np.linspace(0,2*np.pi,1001)[:-1])
    sin_cod = np.sin(np.linspace(0,2*np.pi,1001)[:-1])

    for i in range(len(rvals)):

        np.multiply(ph_basis,freqs[i],out=sph)
        correction = 1-(dt*2*np.pi*freqs[i])**2/24
        tph[:] = sph
        np.subtract(sph,tph,out=sph)
        np.multiply(sph,1000,out=tph,casting='unsafe')
        cph[:] = cos_cod[tph]
        sph[:] = sin_cod[tph]

        tmp[:] = cph[alpha_indices]
        tmp *= weights
        cw1 = np.sum(tmp)
        tmp *= tmp
        cw2 = np.sum(tmp)
        cph *= exp
        cph *= correction
        ce = np.sum(cph)

        # now do same thing for sine
        tmp[:] = sph[alpha_indices]
        tmp *= weights
        sw1 = np.sum(tmp)
        tmp *= tmp
        sw2 = np.sum(tmp)
        sph *= exp
        sph *= correction
        se = np.sum(sph)

        rvals[i] = (cw1-ce)**2/cw2 + (sw1-se)**2/sw2


    return rvals

def power_spectrum_fft(timeseries,dfgoal=None,tweak_exp=False,
        exp_only=False,get_amps=False,exposure_correction=None,
        no_zero_pad=False,maxN=None):
    """ Use FFT to evalute the sums in the maximum likelihood expression.

    This version matches the notation in the paper.

    tweak_exp -- make sure that the mean signal for source and background
        are 0; helps prevent spectral leakage from low frequencies

    get_amps -- if True, return frequencies, real amplitudes, imag. 
        amplitudes, and their uncertainties.  (NB these are the
        *fractional* modulation coefficients.)

    Returns
    -------
    frequencies : in Hz
    P_0 : background fixed power spectrum
    P_1 : background-free spectrum
    P_b : power spectrum of background

    NB: The resulting power spectrum extends only halfway to the Nyquist
    frequency of the input time series because the "upper half" of the
    frequencies are used in the correction needed to correct to Leahy
    normalization.

    NB that the resulting power spectrum is oversampled, at least to the
    nearest convenient power of 2, and of course there are gaps in the
    LAT exposure.  Therefore the oversampling can be e.g. >5x.  Thus care
    is needed if examining the distribution of the PSD, e.g. with a KS
    test, the effective sqrt(N) is smaller than otherwise might seem.

    Use no_zero_pad to return ia critically sampled power spectrum.

    """
    cells = timeseries

    W = cells.weights
    WW = cells.weights2
    S = cells.sexp
    if exposure_correction is not None:
        S = S * exposure_correction
    if tweak_exp:
        S = S*W.sum()/S.sum()
    Wb = cells.counts-W # \bar{W}
    WbWb = cells.counts-2*W+WW
    B = cells.bexp
    if tweak_exp:
        B = B*Wb.sum()/B.sum()

    if maxN is not None:
        W = W[:maxN]
        WW = WW[:maxN]
        Wb = Wb[:maxN]
        WbWb = WbWb[:maxN]
        S = S[:maxN]
        B = B[:maxN]

    if exp_only:
        # this will cause the primary FFT to be of the exposure (with mean
        # subtracted to avoid spectral leage).  The other estimators will be
        # garbage.
        W = 2*S-S.mean()

    # come up with a nice power of 2 for doing the FFT portion
    if dfgoal is None:
        dfgoal = 0.2/cells.tspan()
    nfft = 1./(cells.tsamp()*dfgoal)
    l = 2**(int(np.log2(nfft))+1)
    if no_zero_pad:
        l = len(W)
    zeros = np.zeros(l-len(W))

    # zero pad
    W = np.append(W,zeros)
    WW = np.append(WW,zeros)
    S = np.append(S,zeros)
    Wb = np.append(Wb,zeros)
    WbWb = np.append(WbWb,zeros)
    B = np.append(B,zeros)

    freqs = np.fft.rfftfreq(l)/cells.tsamp()

    # now use FFT to evaluate the various cosine moments, for the
    # maximum likelihood estimators
    f = np.fft.rfft(W-S)[:(l//4+1)]
    WmS_cos = np.real(f)
    WmS_sin = -np.imag(f)

    f = np.fft.rfft(Wb-B)[:(l//4+1)]
    WbmB_cos = np.real(f)
    WbmB_sin = -np.imag(f)

    f = np.fft.rfft(WW)
    S2 = f[0].real
    WW_cos  = 0.5*(S2+np.real(f)[::2])
    WW_sin  = 0.5*(S2-np.real(f)[::2])

    f = np.fft.rfft(WbWb)
    B2 = f[0].real
    WbWb_cos  = 0.5*(B2+np.real(f)[::2])
    WbWb_sin  = 0.5*(B2-np.real(f)[::2])

    f = np.fft.rfft(W-WW)
    SB = f[0].real
    WWb_cos  = 0.5*(SB+np.real(f)[::2])
    WWb_sin  = 0.5*(SB-np.real(f)[::2])

    # Eliminate 0 entries to avoid error messages
    WW_sin[0] = 1.
    WbWb_sin[0] = 1.

    # form non-coupled estimators first
    alpha_cos0 = WmS_cos/WW_cos
    alpha_sin0 = WmS_sin/WW_sin
    beta_cos0 = WbmB_cos/WbWb_cos
    beta_sin0 = WbmB_sin/WbWb_sin
    
    # coupled estimators
    denom_cos = 1./(WW_cos*WbWb_cos-WWb_cos**2)
    denom_sin = 1./(WW_sin*WbWb_sin-WWb_sin**2)
    denom_sin[0] = 0.
    alpha_cos = (WbWb_cos*WmS_cos-WWb_cos*WbmB_cos)*denom_cos
    alpha_sin = (WbWb_sin*WmS_sin-WWb_sin*WbmB_sin)*denom_sin
    beta_cos = (WW_cos*WbmB_cos-WWb_cos*WmS_cos)*denom_cos
    beta_sin = (WW_sin*WbmB_sin-WWb_sin*WmS_sin)*denom_sin

    if get_amps:
        #return freqs[:(l//4+1)],alpha_cos0,alpha_sin0,WW_cos,WW_sin
        return freqs[:(l//4+1)],alpha_cos,alpha_sin,beta_cos,beta_sin,alpha_cos0,alpha_sin0,beta_cos0,beta_sin0

    # for all estimators, the second order correction simply removes half
    # of the value, so that multiplying by 2 gets you back to Leahy.  So
    # let's just stick with the first order!

    # this is the case for non-varying source (alpha(t)=0)
    dlogl_null = beta_cos0*WbmB_cos + beta_sin0*WbmB_sin
    # this is the case for non-varying background (beta(t)=0; P_0 in the paper)
    dlogl_nobg = alpha_cos0*WmS_cos + alpha_sin0*WmS_sin
    # this is the profile likelihood test statistic  (P_s + P_b in the paper)
    dlogl  = alpha_cos*WmS_cos + alpha_sin*WmS_sin + beta_cos*WbmB_cos + beta_sin*WbmB_sin
    if exp_only:
        return freqs[:(l//4+1)],dlogl_nobg
    return freqs[:(l//4+1)],dlogl_nobg,dlogl-dlogl_null,dlogl_null

def power_spectrum_simple(timeseries,get_amps=False):
    """ Follow a similar approach to power_spectrum fft (see docstring)
    but save time by only evaluating the background-fixed quantity.

    Parameters
    ----------
    get_amps : return the cos/sin amplitudes that maximize the likelihood

    Returns
    -------
    freqs,dlogls,[amps] : the frequencies (in Hz), the change in log like
        for the best-fitting amplitude, and if get_amps=True, the
        amplitudes
    """

    W = timeseries.weights
    WW = timeseries.weights2
    S = timeseries.sexp

    l = len(W)

    # now use FFT to evaluate the various cosine moments, for the
    # maximum likelihood estimators
    f = np.fft.rfft(W-S)[:(l//4+1)]
    WmS_cos = np.real(f)
    WmS_sin = -np.imag(f)

    f = np.fft.rfft(WW)
    S2 = f[0].real
    WW_cos  = 0.5*(S2+np.real(f)[::2])
    WW_sin  = 0.5*(S2-np.real(f)[::2])

    # Eliminate 0 entries to avoid error messages
    WW_sin[0] = 1.

    dlogl = WmS_cos**2/WW_cos + WmS_sin**2/WW_sin
    freqs = np.fft.rfftfreq(l)[:(l//4+1)]*(1./timeseries.tsamp())
    if get_amps:
        return freqs,dlogl,(WmS_cos,WmS_sin,WW_cos,WW_sin)
    return freqs,dlogl

def bb_prior_tune(data,tcell,bb_priors=[2,3,4,5,6,7,8,9,10],ntrial=10,
        orbital=False,**cell_kwargs):
    """ Test a range of parameters for an exponential BB prior.
    
    Creating random realizations of the data and run BB algorithm.
    Return values will essentially be the number of partitions for each
    realizations as a function of the parameter value.  This information
    can be used to select an appropriate prior for the real data.
    
    This is very expensive, so note that generally it works pretty well to
    select exp(-gamma) = 1/n_cell.
    """

    rvals = np.empty((len(bb_priors),ntrial),dtype=int)
    for itrial in range(ntrial):

        if orbital:
            cells = data.get_contiguous_exposure_cells(
                    randomize=True,**cell_kwargs)
        elif 'PhaseData' in str(data.__class__):
            cells = data.get_cells(tcell,randomize=True,**cell_kwargs)
        else:
            cells = data.get_cells(tcell=tcell,randomize=True,
                    **cell_kwargs)

        # to save some computation, ignore periodicity in pulsar cases
        # should not be an important effect in determining a good prior
        cll = CellsLogLikelihood(cells)

        for iprior,prior in enumerate(bb_priors):
            segs = cll.do_bb(prior=prior)[0]
            rvals[iprior,itrial] = len(segs)

    return rvals,len(cells)

def plot_bb_prior_results(rvals):
    """ Make a "nice" plot of the result of the output of bb_prior_tune.
    """
    rvals = rvals.astype(int)
    minlength = max(rvals.max()+1,50)
    blah = np.asarray([np.bincount(rvals[i],minlength=minlength) for i in range(rvals.shape[0])])
    pl.clf()
    pl.imshow(blah,interpolation='nearest',aspect='auto')

def get_orbital_modulation(ts,freqs):
    """ Return a time-domain realization of the modulation associated with
        frequency f0.  Include the specified number of harmonics.

    The idea here is to produce an estimate that can be subtracted from the
    baseline expected source counts (from the exposure, with steady rate),
    such that the modulation is now included in the baseline model.

    This is useful for sources with strong modulation, which will leak to
    other frequencies through the window function.  The signal returned by
    this routine can be passed as exposure_correction to 
    power_spectrum_fft.

    Ideally, this would be a full likelihood fit.  Instead, this has some
    of the flaws of the window function, too, but good enough.
    """

    ti = (ts.starts+ts.stops)*0.5
    rvals = np.ones_like(ti)
    y = (ts.weights-ts.sexp)/ts.sexp.mean()
    pows = np.empty(len(freqs))
    for i in range(len(freqs)):
        ph = np.mod((ti-ti[0])*freqs[i],1)*(2*np.pi)
        cph = np.cos(ph)
        sph = np.sin(ph)
        a = np.mean(cph*y)
        b = np.mean(sph*y)
        print(a,b)
        a2 = np.mean(cph**2*ts.weights2)
        b2 = np.mean(sph**2*ts.weights2)
        rvals += 2*(a*cph + b*sph)
        pows[i] = (a**2/a2 + b**2/b2)

    return rvals,pows*ts.sexp.sum()**2/ts.sexp.shape[0]

def plot_raw_lc(rvals,ax=None,scale='linear',min_mjd=None,max_mjd=None,
        ul_color='C1',meas_color='C0',alpha=0.3,ls=' ',ms=3,
        labelsize='large'):
    """ Make a plot of the output of CellsLogLikelihood.get_raw_lightcurve.

    rvals is (N,6) array, with each entry being
        MJDs MJD bin width, flux/alpha, flux_er_lo/uplim, flux_err_hi, TS

    If the entry is an upper limit, flux_err_hi == -1.
    """

    if min_mjd is not None:
        mask = rvals[:,0] >= min_mjd
        rvals = rvals[mask,:]

    if max_mjd is not None:
        mask = rvals[:,0] <= max_mjd
        rvals = rvals[mask,:]

    if ax is None:
        ax = pl.gca()
    ax.set_yscale(scale)

    x,xerr,y,yerrlo,yerrhi,ts = rvals.transpose()
    ul_mask = (yerrhi == -1) & (~np.isnan(yerrhi))

    # TODO -- obviously shouldn't have negative values, so need to fix that
    # I guess this is from cases where the confidence limit can't be
    # satisfied.
    yerrlo[yerrlo <= 0] = y[yerrlo <= 0]

    # Plot the upper limits
    x,xerr,y,yerrlo,yerrhi,ts = rvals[ul_mask].transpose()
    ax.errorbar(x=x,y=y,xerr=xerr,
            yerr=0.2*(1 if scale=='linear' else y),
            uplims=True,marker=None,color=ul_color,alpha=alpha,ls=ls,ms=ms)

    # Plot the measurements
    x,xerr,y,yerrlo,yerrhi,ts = rvals[~ul_mask].transpose()
    ax.errorbar(x=x,y=y,xerr=xerr,yerr=[yerrlo,yerrhi],
            marker='o',color=meas_color,alpha=alpha,ls=ls,ms=ms)

    ax.set_xlabel('MJD',size=labelsize)
    ax.set_ylabel('Relative Flux',size=labelsize)

def plot_bb_lc(rvals,ax=None,scale='linear',min_mjd=None,max_mjd=None,
        ul_color='C3',meas_color='C3',alpha=0.8,ls=' ',ms=3,
        labelsize='large'):
    """ Make a plot of the output of CellsLogLikelihood.get_bb_lightcurve.

    rvals is (N,6) array, with each entry being
        MJDs MJD bin width, flux/alpha, flux_er_lo/uplim, flux_err_hi, TS

    If the entry is an upper limit, flux_err_hi == -1.
    """

    import inspect
    sig = inspect.signature(plot_raw_lc)
    l = locals()
    args = [l[k] for k in sig.parameters.keys()]
    plot_raw_lc(*args)

    #plot_raw_lc(rvals,ax=ax,scale=scale,min_mjd=min_mjd,max_mjd=max_mjd,
    #    ul_color=ul_color,meas_color=meas_color,alpha=alpha,ls=ls,ms=ms)

def plot_both_lc(rvals_raw,rvals_bb,ax=None):
    """ Make a standard plot with the raw and BB light curves shown."""
    plot_raw_lc(rvals_raw,ax=ax)
    plot_bb_lc(rvals_bb,ax=ax)
