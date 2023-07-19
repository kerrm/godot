# for some reason this needs to be first with new conda install.  gross.
from __future__ import print_function
import py_exposure_p8

import numpy as np
import pylab as pl
from collections import deque
from scipy.integrate import simps,cumtrapz
from scipy.optimize import fmin,fsolve,fmin_tnc,brentq
from scipy.interpolate import interp1d
from scipy.stats import chi2
from astropy.io import fits

import bary

# MET bounds for 8-year data set used for FL8Y and 4FGL
t0_8year = 239557007.6
t1_8year = 491999980.6

def met2mjd(times,mjdref=51910+7.428703703703703e-4):
    times = np.asarray(times,dtype=np.float128)
    return times*(1./86400)+mjdref

def mjd2met(times,mjdref=51910+7.428703703703703e-4):
    times = (np.asarray(times,dtype=np.float128)-mjdref)*86400
    return times


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
    ti = np.concatenate([c.we for c in cells])
    exp = np.sum((c.exp for c in cells))
    if not np.all(np.asarray([c.SonB for c in cells])==cells[0].SonB):
        raise Exception('Cells do not all have same source flux!')
    return Cell(cells[0].tstart,cells[-1].tstop,exp,ti,we,cells[0].SonB)

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
            counts,weights,weights2,
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
        self.alt_starts = alt_starts
        self.alt_stops = alt_stops
        self.timesys = timesys
        ## exposure mask
        self.minimum_exposure = minimum_exposure
        mask = ~(exp > minimum_exposure)
        #print('exposure filter %d/%d:'%(mask.sum(),len(mask)))
        self.exp[mask] = 0
        self.sexp[mask] = 0
        self.bexp[mask] = 0
        self.counts[mask] = 0
        self.weights[mask] = 0
        self.weights2[mask] = 0

    def tsamp(self):
        # TODO -- perhaps put in a contiguity check
        return self.stops[0]-self.starts[0]

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


class CellLogLikelihood(object):

    def __init__(self,cell):
        self.cell = cell
        self.ti = cell.ti
        self.we = cell.we
        self.iwe = 1-self.we
        self.S = cell.exp
        self.B = self.S/cell.SonB
        self._tmp1 = np.empty_like(self.we)
        self._tmp2 = np.empty_like(self.we)
        self._last_beta = 0.

    def log_likelihood(self,alpha):
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
        alpha -= 1
        beta -= 1
        S,B = self.S,self.B
        t1 = np.multiply(self.we,alpha-beta,out=self._tmp1)
        t1 += 1+beta
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

            # I'm not sure if it's best to keep to a default guess or to
            # try it at the non-profile best-fit.
            #if guess == 1:
                #guess = self.get_max(profile_background=False)
            #if beta == 1:
                #beta = self.get_beta_max(guess)
            # NB -- removed the scale below as it seemed to work better
            # without it.  In other words, the scale is already ~1!
            # test TNC method
            rvals,nfeval,rc = fmin_tnc(self.fmin_tnc_func,[guess,beta],
                    bounds=[[0,None],[0,None]],disp=0,ftol=1e-3)
            # this is one possible way to check for inconsistency in max
            #if (guess == 1) and (rvals[0] > 10):
                #rvals,nfeval,rc = fmin_tnc(self.fmin_tnc_func,
                        #[rvals[0],beta],
                        #bounds=[[0,None],[0,None]],disp=0,ftol=1e-3)

            if not((rc < 0) or (rc > 2)):
                if (guess == 0) and (rvals[0] > 5e-2):
                    print('Warning, possible inconsistency.  Guess was 0, best fit value %.5g.'%(rvals[0]),'beta=',beta)
                return rvals

            # try a small grid to seed a search
            grid = np.asarray([0,0.1,0.3,0.5,1.0,2.0,5.0,10.0])
            cogrid = np.asarray([self.log_profile_likelihood(x) for x in grid])
            newguess = grid[np.argmax(cogrid)]
            newbeta = max(0.1,self.get_beta_max(newguess))

            rvals,nfeval,rc = fmin_tnc(self.fmin_tnc_func,
                    [newguess,newbeta],
                    bounds=[[0,None],[0,None]],disp=0,ftol=1e-3)
            if not((rc < 0) or (rc > 2)):
                return rvals
            else:
                print('Warning, trouble locating maximum with profile_background!  Results for this interval may be unreliable.')
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
        #display = False
        alpha = alpha-1
        guess -= 1
        S,B = self.S,self.B
        w = self.we
        iw = self.iwe
        t,t2 = self._tmp1,self._tmp2

        # check that the maximum isn't at 0 (-1) with derivative
        t2[:] = 1+alpha*w
        t[:] = iw/(t2-iw)
        if (np.sum(t)-B) < 0:
            return 0
        else:
            b = guess
        #if display:
        #    print '0:',b+1

        # on first iteration, don't let it go to 0
        t[:] = iw/(t2+b*iw)
        f1 = np.sum(t)-B
        t *= t
        f2 = np.sum(t) # will include sign below
        b = b + f1/f2
        #if display:
        #    print '1:',b+1
        b = max(0.2-1,b)
        #if b < 0.2-1:
            #b = 0.2-1
        #if display:
        #    print '1p:',b+1

        # second iteration more relaxed
        t[:] = iw/(t2+b*iw)
        f1 = np.sum(t)-B
        t *= t
        f2 = np.sum(t) # will include sign below
        b = b + f1/f2
        #if display:
        #    print '2:',b+1
        b = max(0.05-1,b)
        #if b < 0.05-1:
            #b = 0.05-1
        #if display:
        #    print '2p:',b+1

        # last NR iteration allow 0
        # second iteration more relaxed
        t[:] = iw/(t2+b*iw)
        f1 = np.sum(t)-B
        t *= t
        f2 = np.sum(t) # will include sign below
        b = b + f1/f2
        #if display:
        #    print '3:',b+1
        #if b < 0-1:
            #b = 0.02-1
        b = max(0.02-1,b)
        #if display:
        #    print '3p:',b+1

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
        #if display:
        #    print '4:',newb+1,b+1
        if abs(newb-b) > 10:
            blast = b = 2*b+1
        else:
            blast = b = newb
        #if display:
        #    print '4p:',b+1

        # now do a last Hally iteration
        t[:] = iw/(t2+b*iw)
        f1 = np.sum(t)-B
        t *= t
        f2 = np.sum(t) # will include sign below
        t *= iw/(t2+b*iw)
        f3 = 2*np.sum(t)
        b = max(0-1,b + 2*f1*f2/(2*f2*f2-f1*f3))
        #if display:
        #    print '5:',b+1

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
        if not include_zero:
            # ditto, lower side
            t[:] = aopt*we + iw
            # use Taylor approximation to get initial guess
            f2 = np.abs(np.sum((we/t)**2))
            amin = aopt - np.sqrt(2*dlogl/f2)
            # do a few NR iterations
            for i in range(5):
                t[:] = amin*we+iw
                f0 = np.log(t).sum()-amin*S+dlogl-llmax
                f1 = np.sum(we/t)-S
                amin = amin - f0/f1
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
        func = self.log_profile_likelihood if profile_background else self.log_likelihood
        if abs(dom[amax]-aopt) > 0.1: # somewhat ad hoc
            # re-optimize
            aopt = self.get_max(guess=dom[amax],
                    profile_background=profile_background)
            if profile_background:
                aopt = aopt[0]
            if abs(dom[amax]-aopt) > 0.1: # somewhat ad hoc
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
    def __init__(self,cells,profile_background=False):

        # construct a sampled log likelihoods for each cell
        self.clls = [CellLogLikelihood(x) for x in cells]
        npt = 200
        self._cod = np.empty([len(cells),npt])
        self._dom = np.empty([len(cells),npt])
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

    def get_lightcurve(self,tsmin=4,plot_years=False,plot_phase=False,
            get_ts=False):
        """ Return a flux density light curve for the raw cells.
        """

        plot_phase = plot_phase or isinstance(self,PhaseCellsLogLikelihood)

        # time, terr, yval, yerrlo,yerrhi; yerrhi=-1 if upper limit
        rvals = np.empty([len(self.clls),5])
        all_ts = np.empty(len(self.clls))
        for icll,cll in enumerate(self.clls):
            if cll.S==0:
                rvals[icll] = np.nan
            tmid = cll.cell.get_tmid()
            if plot_phase:
                terr = (tmid-cll.cell.tstart)
            else:
                terr = (tmid-cll.cell.tstart)/86400
                tmid = met2mjd(tmid)
                if plot_years:
                    tmid = (tmid-54832)/365 + 2009 
                    terr *= 1./365
            aopt,ts,xconf = self.get_flux(icll,conf=[0.16,0.84])
            ul = ts <= tsmin
            if ul:
                rvals[icll] = tmid,terr,xconf[1],0,-1
            else:
                rvals[icll] = tmid,terr,aopt,aopt-xconf[0],xconf[1]-aopt
            all_ts[icll] = ts

        if get_ts:
            return rvals,all_ts
        return rvals

    def plot_cells_bb(self,tsmin=4,fignum=2,clear=True,color='C3',
            plot_raw_cells=True,bb_prior=4,plot_years=False,
            no_bb=False,no_ts=True,log_scale=False,
            plot_phase=False,ax=None,labelsize='large'):
        """ Generate a plot showing fluxes both from the raw cells and from
            a Bayesian blocks partition which is computed using the input
            prior.

            Parameters:
            tsmin -- plot upper limits if cells/partition TS is <tsmin
            fignum -- matplotlib figure number to use
            clear -- clear previous matplotlib figure
            color -- [WARNING -- NOT USED?]
            plot_raw_cells -- if False, do not plot underyling  fluxes
            bb_prior -- exponential prior for BB algorithm
            plot_years -- if True, scale data to years; default is days
            no_bb -- do not run/plot/return Bayesian Blocks results
            no_ts -- do not include TS values in return
            log_scale -- set y-axis scale to log
            plot_phase -- interpret times as phase [0,1) instead
            ax -- use provided instance to plot on
        """

        # NB might want to use a CellsLogLikelihood to avoid overhead of 3x
        # size on BB computation
        plot_phase = plot_phase or isinstance(self,PhaseCellsLogLikelihood)

        if ax is None:
            pl.figure(fignum)
            if clear:
                pl.clf()
            ax = pl.gca()

        if log_scale:
            ax.set_yscale('log')
        if plot_raw_cells:
            # time, terr, yval, yerrlo,yerrhi; yerrhi=-1 if upper limit
            rvals = np.empty([len(self.clls),5 if no_ts else 6])
            for icll,cll in enumerate(self.clls):
                if cll.S==0:
                    rvals[icll] = np.nan
                tmid = cll.cell.get_tmid()
                if plot_phase:
                    terr = (tmid-cll.cell.tstart)
                else:
                    terr = (tmid-cll.cell.tstart)/86400
                    tmid = met2mjd(tmid)
                    if plot_years:
                        tmid = (tmid-54832)/365 + 2009 
                        terr *= 1./365
                aopt,ts,xconf = self.get_flux(icll,conf=[0.16,0.84])
                if ts <= tsmin:
                    if no_ts:
                        rvals[icll] = tmid,terr,xconf[1],0,-1
                    else:
                        rvals[icll] = tmid,terr,xconf[1],0,-1,ts
                else:
                    if no_ts:
                        rvals[icll] = tmid,terr,aopt,aopt-xconf[0],xconf[1]-aopt
                    else:
                        rvals[icll] = tmid,terr,aopt,aopt-xconf[0],xconf[1]-aopt,ts
            ul_mask = (rvals[:,-1] == -1) & (~np.isnan(rvals[:,-1]))
            t = rvals[ul_mask].transpose()
            ax.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,
                    marker=None,color='C0',alpha=0.2,ls=' ',ms=3)
            t = rvals[~ul_mask].transpose()
            ax.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',
                    color='C0',alpha=0.2,ls=' ',ms=3)
        else:
            rvals = None

        # now, do same for Bayesian blocks
        if not no_bb:
            bb_idx,bb_ts,var_ts,var_dof,fitness = self.do_bb(prior=bb_prior)
            print(var_ts,var_dof)
            print('Variability significance: ',chi2.sf(var_ts,var_dof))
            bb_idx = np.append(bb_idx,len(self.cells))
            rvals_bb = np.empty([len(bb_idx)-1,5 if no_ts else 6])
            for ibb,(start,stop) in enumerate(zip(bb_idx[:-1],bb_idx[1:])):
                cells = cell_from_cells(self.cells[start:stop])
                cll = CellLogLikelihood(cells)
                if cll.S==0:
                    rvals_bb[ibb] = np.nan
                    continue
                tmid = cll.cell.get_tmid()
                if plot_phase:
                    terr = (tmid-cll.cell.tstart)
                else:
                    terr = (tmid-cll.cell.tstart)/86400
                    tmid = met2mjd(tmid)
                    if plot_years:
                        tmid = (tmid-54832)/365 + 2009 
                        terr *= 1./365
                aopt,ts,xconf = cll.get_flux(conf=[0.16,0.84],
                        profile_background=self.profile_background)
                if ts <= tsmin:
                    if no_ts:
                        rvals_bb[ibb] = tmid,terr,xconf[1],0,-1
                    else:
                        rvals_bb[ibb] = tmid,terr,xconf[1],0,-1,ts
                else:
                    if no_ts:
                        rvals_bb[ibb] = tmid,terr,aopt,aopt-xconf[0],xconf[1]-aopt
                    else:
                        rvals_bb[ibb] = tmid,terr,aopt,aopt-xconf[0],xconf[1]-aopt,ts

            ul_mask = (rvals_bb[:,-1] == -1) & (~np.isnan(rvals_bb[:,-1]))
            t = rvals_bb[ul_mask].transpose()
            ax.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,
                    color='C3',alpha=0.8,ls=' ',ms=3)
            t = rvals_bb[~ul_mask].transpose()
            ax.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C3',
                    alpha=0.8,ls=' ',ms=3)
        else:
            rvals_bb=None

        if plot_phase:
            ax.set_xlabel('Pulse Phase',size=labelsize)
            ax.axis([0,1,pl.axis()[2],pl.axis()[3]])
        elif plot_years:
            ax.set_xlabel('Year',size=labelsize)
        else:
            ax.set_xlabel('MJD',size=labelsize)
        ax.set_ylabel('Relative Flux Density',size=labelsize)
        return rvals,rvals_bb

    def get_bb_lightcurve(self,tsmin=4,plot_years=False,plot_phase=False,
            bb_prior=8):
        """ Return a flux density light curve for the raw cells.
        """

        plot_phase = plot_phase or isinstance(self,PhaseCellsLogLikelihood)

        # now, do same for Bayesian blocks
        bb_idx,bb_ts,var_ts,var_dof,fitness = self.do_bb(prior=bb_prior)
        print(var_ts,var_dof)
        print('Variability significance: ',chi2.sf(var_ts,var_dof))
        bb_idx = np.append(bb_idx,len(self.cells))
        rvals_bb = np.empty([len(bb_idx)-1,5])
        for ibb,(start,stop) in enumerate(zip(bb_idx[:-1],bb_idx[1:])):
            cells = cell_from_cells(self.cells[start:stop])
            cll = CellLogLikelihood(cells)
            if cll.S==0:
                rvals_bb[ibb] = np.nan
                continue
            tmid = cll.cell.get_tmid()
            if plot_phase:
                terr = (tmid-cll.cell.tstart)
            else:
                terr = (tmid-cll.cell.tstart)/86400
                tmid = met2mjd(tmid)
                if plot_years:
                    tmid = (tmid-54832)/365 + 2009 
                    terr *= 1./365
            aopt,ts,xconf = cll.get_flux(conf=[0.16,0.84],
                    profile_background=self.profile_background)
            if ts <= tsmin:
                rvals_bb[ibb] = tmid,terr,xconf[1],0,-1
            else:
                rvals_bb[ibb] = tmid,terr,aopt,aopt-xconf[0],xconf[1]-aopt

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
            zenith_cut=90,minimum_exposure=3e4,use_phi=True,
            base_spectrum=None,use_weights_for_exposure=False,
            weight_cut=1,max_radius=None,bary_ft1files=None,
            tstart=None,tstop=None,apply_8year_scale=False):
        """ The FT1 files, FT2 files;
            ra, dec of source (deg)
            weight_cut -- fraction of source photons to retain
            max_radius -- maximum photon separation from source [deg]
            bary_ft1files -- an optional mirrored set of FT1 files with
                photon timestamps in the barycentric reference frame
            tstart, tstop -- notionally MET, but will attempt to convert
                from MJD if < 100,000.

        Generally, the times for everything should be topocentric.
        However, if there is a need to search for short period signals,
        then barycentric times should be used.  Exposure calculations must
        still be topocentric, so the easiest solution is simply to provide
        both files.
        """
        self.ft1files = ft1files
        self.ft2files = ft2files
        self.max_radius = max_radius
        self.ra,self.dec = ra,dec
        self._barycon = None

        if tstart is not None:
            if tstart < 100000:
                tstart = mjd2met(tstart)
        if tstop is not None:
            if tstop < 100000:
                tstop = mjd2met(tstop)
        
        lt = py_exposure_p8.Livetime(ft2files,ft1files,
                tstart=tstart,tstop=tstop)
        mask,pcosines,acosines = lt.get_cosines(
                np.radians(ra),np.radians(dec),
                theta_cut=0.4,zenith_cut=np.cos(np.radians(zenith_cut)),
                get_phi=use_phi)
        phi = np.arccos(acosines) if use_phi else None
        if use_weights_for_exposure:
            base_spectrum = None
        aeff,self.total_exposure_edom,self.total_exposure = self._get_weighted_aeff(pcosines,phi,base_spectrum)
        exposure = np.zeros(len(mask))
        exposure[mask] = aeff*lt.LIVETIME[mask]
        # just get rid of any odd bins -- this cut corresponds to roughly
        # 30 seconds of exposure at the edge of the FoV; in practice it
        # doesn't remove too many photons
        self.minimum_exposure = minimum_exposure
        full_mask = mask & (exposure>minimum_exposure)
        exposure[~full_mask] = 0

        self.TSTART = lt.START[full_mask]
        self.TSTOP = lt.STOP[full_mask]
        self.exposure = exposure[full_mask]
        self.cexposure = np.cumsum(self.exposure)

        data,self.timeref,photon_idx = self._load_photons(
                ft1files,weight_col,self.TSTART[0],self.TSTOP[-1],
                max_radius=max_radius)
        ti = data[0]
        if self.timeref=='SOLARSYSTEM':
            print('WARNING!!!!  Barycentric data not accurately treated')
        
        # this is a problem if the data are barycentered
        event_idx = np.searchsorted(lt.STOP,ti)
        # TODO add a check for FT2 files that are shorter than observation
        event_mask = self.event_mask = exposure[event_idx] > 0
        event_mask &= lt.get_gti_mask(ti)

        data = [d[event_mask].copy() for d in data]
        self.ti = data[0]
        self.we = data[1]
        self.other_data = data[2:]

        if bary_ft1files is not None:
            data,timeref = self._load_photons(bary_ft1files,weight_col,
                    None,None,max_radius=max_radius,no_filter=True)
            self.bary_ti = (data[0][photon_idx][event_mask]).copy()
        else:
            self.bary_ti = None

        if use_weights_for_exposure:
            print('beginning exposure refinement')
            # do a refinement of exposure calculation
            aeff,self.total_exposure_edom,self.total_exposure = self._get_weighted_aeff(pcosines,phi,base_spectrum=None,use_event_weights=True,livetime=lt.LIVETIME[mask])
            print('finished exposure refinement')


        # do another sort into the non-zero times
        event_idx = self.event_idx = np.searchsorted(self.TSTOP,self.ti)

        # NEEDED?
        #print 'beginning photon exposure'
        #self.photon_exposure = self.get_exposure(self.ti)
        #print 'ending photon exposure'

        self.weight_cut = weight_cut
        if weight_cut < 1:
            swe = np.sort(self.we)
            t = np.cumsum(swe)
            wmin = swe[np.searchsorted(t,(1-weight_cut)*t[-1])]
            mask = self.we >= wmin
            self.we = self.we[mask]
            self.ti = self.ti[mask]
            if self.bary_ti is not None:
                self.bary_ti = self.bary_ti[mask]
            # not sure if want to do this, but this preserves source flux
            self.exposure *= self.weight_cut

        if apply_8year_scale:
            # reweight everything so that variable sources whose weights
            # are computed with an 8-year model will have correct scaling
            # to give a mean flux of 1 over the full data range
            mask = self.ti < t1_8year
            emask = self.TSTOP <= t1_8year
            s8 = self.we[mask].sum()/self.exposure[emask].sum()
            stot = self.we.sum()/self.exposure.sum()
            new_alpha = stot/s8
            print('Rescaling with alpha=%.2f.'%(new_alpha))
            self.we = new_alpha*self.we/(new_alpha*self.we+(1-self.we))

        self.E = np.sum(self.exposure)
        self.S = np.sum(self.we)
        self.B = len(self.we)-self.S

    def __setstate__(self,state):
        if not '_barycon' in state:
            state['_barycon'] = None
        self.__dict__.update(state)

    def _get_weighted_aeff(self,pcosines,phi,base_spectrum=None,
            use_event_weights=False,livetime=None):
        ea = py_exposure_p8.EffectiveArea(
                irf='P8R2_SOURCE_V6',use_phidep=phi is not None)
        if base_spectrum is not None:
            edom = np.logspace(2,5,25)
            wts = base_spectrum(edom)
            total_exposure = np.empty_like(edom)
            wts = base_spectrum(edom)
            rvals = np.empty([len(edom),len(pcosines)])
            for i,(en,wt) in enumerate(zip(edom,wts)):
                faeff,baeff = ea([en],pcosines,phi=phi)
                rvals[i] = (faeff+baeff)*wt
                total_exposure[i] = rvals[i].sum()/wt
            aeff = simps(rvals,edom,axis=0)/simps(wts,edom)
        elif use_event_weights:
            # TODO -- this is a real mess.  Make this more consistent
            # or else break it out.  On the bright sde, it seems to work!
            # also, this can be speeded up substantially by binning in cos
            # theta, at least for calculating the exposure.
            if livetime is None:
                raise ValueError('Must provide livetime argument.')
            try:
                en = self.other_data[self.other_data_cols.index('energy')]
            except ValueError:
                return self._get_weighted_aeff(pcosines,phi,base_spectrum=base_spectrum,use_event_weights=False)
            edom = np.logspace(2,5,25)
            wcts = np.histogram(en,weights=self.we,bins=edom)[0]
            ledom = np.log10(edom)
            edom = 10**(0.5*(ledom[:-1]+ledom[1:]))
            rvals = np.empty([len(edom),len(pcosines)])
            wts = np.empty(len(edom))
            total_exposure = np.empty_like(edom)
            for i in range(len(edom)):
                faeff,baeff = ea([edom[i]],pcosines,phi=phi)
                faeff += baeff
                total_exposure[i] = np.sum(faeff*livetime)
                wts[i] = wcts[i]/total_exposure[i]
                rvals[i] = (faeff+baeff)*wts[i]
            # cache the weights for double checking
            self._exposure_weights = wts
            aeff = simps(rvals,edom,axis=0)/simps(wts,edom)
        else:
            edom = [1000]
            faeff,baeff = ea([1000],pcosines,phi=phi)
            aeff = faeff+baeff
            total_exposure = [aeff.sum()]
        return aeff,edom,total_exposure

    def _load_photons(self,ft1files,weight_col,tstart,tstop,
            max_radius=None,time_col='time',no_filter=False):
        cols = [time_col,weight_col,'energy']
        deques = [deque() for col in cols]
        for ift1,ft1 in enumerate(ft1files):
            f = fits.open(ft1)
            if ift1 == 0:
                timeref = f['events'].header['timeref']
            else:
                if f['events'].header['timeref'] != timeref:
                    raise Exception('Different time systems!')
            # sanity check for weights computation
            try:
                f['events'].data.field(weight_col)
            except KeyError:
                print(f'FT1 file {ft1} did not have weights column {weight_col}!  Skipping.')
                continue
            if max_radius is not None:
                ra = np.radians(f['events'].data.field('ra'))
                dec = np.radians(f['events'].data.field('dec'))
                cdec,sdec = np.cos(dec),np.sin(dec)
                cdec_src = np.cos(np.radians(self.dec))
                sdec_src = np.sin(np.radians(self.dec))
        # cosine(polar angle) of source in S/C system
                cosines  = cdec_src*np.cos(dec)*np.cos(ra-np.radians(self.ra)) + sdec_src*np.sin(dec)
                mask = cosines >= np.cos(np.radians(max_radius))
                print('keeping %d/%d photons for radius cut'%(mask.sum(),len(mask)))
            else:
                mask = slice(0,f['events'].header['naxis2'])
            for c,d in zip(cols,deques):
                d.append(f[1].data.field(c)[mask])

        # sort everything on time (in case files unordered)
        if no_filter:
            data = [(np.concatenate(d)).copy() for d in deques]
            return data,timeref

        ti = np.concatenate(deques[0])
        a = np.argsort(ti)
        #ti = ti[a] # I think this was incorrect
        # the argsort mask order 
        if (tstart is not None) and (tstop is not None):
            print('applying time cut')
            tstart = tstart or 0
            tstop = tstop or 999999999
            idx = a[(ti >= tstart) & (ti <= tstop)]
        else:
            idx = a
        data = [(np.concatenate(d)[idx]).copy() for d in deques]
        self.other_data_cols = cols[2:]
        return data,timeref,idx

    def _bary2topo(self,bary_times,quiet=True):
        if self._barycon is not None:
            print('Using cached interpolator.')
            return self._barycon(bary_times)
        # Ignores Fermi position, so just gives barycenter time to
        # +/- 20ms accuracy.
        # Generate a set of knots in topocentric time, currently 3hr.  
        knot_space = 3600*3
        topo_knots = np.arange(
                self.TSTART[0]-2*knot_space,self.TSTOP[-1]+2*knot_space+1,
                knot_space)
        if (not quiet):
            print('Warning!  This formulation does not account for travel time around the earth; all conversions done for geocenter.')
        if not quiet:
            print('beginning barycenter for topocentric knots')
        bary_knots = bary.met2tdb(topo_knots,self.ra,self.dec)
        if not quiet:
            print('ending barycenter for topocentric knots')
        self._barycon = interp1d(bary_knots,topo_knots,bounds_error=True)
        return self._barycon(bary_times)

    def get_exposure(self,times):
        """ Return the cumulative exposure at the given times.
        """
        TSTART,TSTOP = self.TSTART,self.TSTOP
        idx = np.searchsorted(TSTOP,times)
        # if time falls between exposures, the idx will be of the following
        # livetime entry, but that's OK, because we'll just mask on the
        # negative fraction; ditto for something that starts before
        frac = (times -TSTART[idx])/(TSTOP[idx]-TSTART[idx])
        np.clip(frac,0,1,out=frac)
        return self.cexposure[idx]-self.exposure[idx]*(1-frac)

    def get_contiguous_exposures(self,tstart=None,tstop=None,
            max_interval=10):
        """ Return those intervals where the exposure is uninterrupted.
        This will typically be an orbit, or possibly two portions of an
        orbit if there is an SAA passage.

        max_interval -- maximum acceptable break in exposure for a
            contiguous interval [10s]
        """

        t0s = self.TSTART
        t1s = self.TSTOP
        if tstart is not None:
            idx = np.searchsorted(t1s,tstart)
            t0s = t0s[idx:].copy()
            t1s = t1s[idx:].copy()
            if tstart < t0s[0]:
                tstart = t0s[0]
            else:
                t0s[0] = tstart
        if tstop is not None:
            idx = np.searchsorted(t1s,tstop)
            t0s = t0s[:idx].copy()
            t1s = t1s[:idx].copy()
            if tstop < t0s[-1]:
                t0s = t0s[:-1]
                t1s = t1s[:-1]
                tstop = t1s[-1]
            else:
                t1s[-1] = tstop

        break_mask = (t0s[1:]-t1s[:-1])>max_interval
        break_starts = t1s[:-1][break_mask]
        break_stops = t0s[1:][break_mask]
        # now assumble the complement
        good_starts = np.empty(len(break_starts)+1)
        good_stops = np.empty_like(good_starts)
        good_starts[0] = t0s[0]
        good_starts[1:] = break_stops
        good_stops[-1] = t1s[-1]
        good_stops[:-1] = break_starts
        return good_starts,good_stops

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
        exposure = self.get_exposure(times)

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
        exposure = self.get_exposure(cell_stops) - self.get_exposure(cell_starts)

        # there should now be a 1-1 mapping between tstart, tstop,
        # photon times, photon weights, and exposures
        return list(map(Cell,cell_starts,cell_stops,exposure*(self.S/self.E),ti,self.we[event_idx0:event_idx1],[self.S/self.B]*len(cell_starts)))

    def get_cells(self,tstart=None,tstop=None,tcell=None,
            snap_edges_to_exposure=False,trim_zero_exposure=True,
            time_series_only=False,use_barycenter=True,
            randomize=False,scale=None,
            scale_series=None,exposure_scaler=None,
            minimum_exposure=3e4,minimum_fractional_exposure=0,
            quiet=False):
        """ Return the starts, stops, exposures, and photon data between
            tstart and tstop.  If tcell is specified, bin data into cells
            of length tcell (s).  Otherwise, return photon-based cells.

            snap_edges_to_exposure -- move the edges of the returned cells
                to align with the resolution of the underlying FT2 file; 
                this basically cuts out dead time, and is potentially
                useful if tcell is a few hours or less. [NOT IMPLEMENTED]

            trim_zero_exposure -- remove Cells that have 0 exposure.  good
                for things like Bayesian Block analyses, VERY BAD for
                frequency-domain analyses!

            time_series_only -- don't return Cells, but a list of time
                series: starts, stops, exposure, cts, sum(weights), 
                sum(weights^2)

            use_barycenter -- interpret tcell in barycenter frame, so
                generate a set of nonuniform edges in the topocenter and
                use this for binning/exposure calculation; NB that in
                general one would *not* use barycentered event times
                in this case.  ALSO note that this calculation is approx.
                because it (almost always) assumes that the S/C is at the
                geocenter, so will be out by up to 40ms.

            randomize -- shuffle times and weights so that they follow
                the exposure but lose all time ordering; useful for
                exploring null cases

            scale -- apply a single rescaling weight

            scale_series -- a series of rescaling weights; the format
                should be passed as a tuple (edges,scales) with the edges
                giving the boundaries of the time interval of scale
                validity; these should be contiguous.  Times in MET.

                This is useful for analyses like Cygnus X-3, there there
                is overall "slow" source variation and a fast (orbital)
                periodicity.  Scaling so that the weights account for the
                slow variation (enhancing the times when it is on) improves
                the periodicity sensitivity.

            minimum_exposure -- this is the minimum exposure, scaled by
                tcell/30.  Only applied if time_series_only.
                [To be confirmed: possibly also just set to 0.  I don't
                see why we would necessarily want to add another cut here.]

            minimum_fractional_exposure -- reject cells whose exposure
                is less than this fraction of the mean exposure of all of
                the cells.  Only valid when tcell is specified.  Use with
                care if using short (e.g. <1d) tcell, as the intrinsic
                exposure variation becomes large.

            exposure_scaler -- a function which will take MET and give
                a scaling factor.  This is applied to each exposure
                interval, and the weights are re-distributed according to
                the re-scaled exposure.  This allows for the injection of
                a signal into the data.  (NB -- this is also removes any
                original signal because the weights are fully shuffled
                according to the new exposure.  A uniform rescale is
                equivalent to randomize=True.)

            TODO -- implement a "use FT2 cells" feature, and a "use orbits"
            feature

        """
        ft1_is_bary = self.timeref == 'SOLARSYSTEM'

        if tstart is None:
            tstart = self.TSTART[0]
        if tstart < 100000:
            tstart = mjd2met(tstart)
        if tstart < self.TSTART[0]:
            print('Warning: Start time precedes start of exposure.')
            print('Will clip to MET=%.2f.'%(self.TSTART[0]))
            tstart = self.TSTART[0]

        if tstop is None:
            tstop = self.TSTOP[-1]
        if tstop < 100000:
            tstop = mjd2met(tstop)
        if tstop > self.TSTOP[-1]:
            print('Warning: Stop time follows end of exposure.')
            print('Will clip to MET=%.2f.'%(self.TSTOP[-1]))
            tstop = self.TSTOP[-1]

        if scale_series is not None:
            # make scale_series consistent with start and stop time
            tstart = max(scale_series[0][0],tstart)
            tstop = min(scale_series[0][-1],tstop)
            mask = (scale_series[0][1:] > tstart) & (scale_series[0][:-1] < tstop)
            new_edges = scale_series[0][:-1][mask]
            new_edges = np.append(new_edges,scale_series[0][1:][mask][-1])
            new_scales = scale_series[1][mask]
            scale_series = [new_edges,new_scales]

        if use_barycenter:
            tstart,tstop = bary.met2tdb([tstart,tstop],self.ra,self.dec)

        if tcell is None:
            return self.get_photon_cells(tstart,tstop)

        ncell = int((tstop-tstart)/tcell)
        if ncell == 0:
            tcell = tstop-tstart
            ncell = 1

        edges = tstart + np.arange(ncell+1)*tcell
        if use_barycenter:
            topo_edges = self._bary2topo(edges)
            bary_edges = edges
        else:
            topo_edges = edges

        # always use topocentric times to manage the exposure calculation
        self._topo_edges = topo_edges.copy()
        cexp = self.get_exposure(topo_edges)
        exp = (cexp[1:] - cexp[:-1])
        if snap_edges_to_exposure:
            raise NotImplementedError
        else:
            if ft1_is_bary:
                starts,stops = bary_edges[:-1],bary_edges[1:]
            else:
                starts,stops = topo_edges[:-1],topo_edges[1:]

        # trim off any events that come outside of first/last cell
        istart,istop = np.searchsorted(self.ti,[starts[0],stops[-1]])
        times = self.ti[istart:istop]
        weights = self.we[istart:istop]

        if minimum_fractional_exposure > 0:
            frac_exp = exp/exp[exp>0].mean()
            exp[frac_exp < minimum_fractional_exposure] = 0

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
        else:
            exposure_mask = slice(0,len(starts))

        if randomize or (exposure_scaler is not None):
            scales = 1
            if exposure_scaler is not None:
                scales = exposure_scaler(starts,stops)
            # NOT SURE how consistent this is with e.g.
            # minimum_fractional exposure, use with care
            cexp = np.cumsum(exp*scales)
            cexp *= 1./cexp[-1]
            indices = np.searchsorted(cexp,np.random.rand(len(times)))
            a = np.argsort(indices)
            times = times[a]
            weights = weights[a]
            event_idx = indices[a]
        else:
            event_idx = np.searchsorted(stops,times)

        nweights = np.bincount(event_idx,minlength=len(starts))
        self._exp = exp.copy()

        # rescale the weights
        if scale_series is not None:
            # search right edges
            edges,scales = scale_series
            right_edges = edges[1:]
            # adjust weights
            W0 = weights.sum()
            Wb0 = (1-weights).sum()
            idx = np.searchsorted(right_edges,times)
            scale = scales[idx]
            weights = scale*weights/(scale*weights+(1-weights))
            W1 = weights.sum()
            Wb1 = (1-weights).sum()
            # adjust exposure
            idx = np.searchsorted(right_edges,stops)
            sexp = (exp * scales[idx])*(self.S/self.E)
            # set overall scale; for now, assume that the scale series
            # is averaged to 1;  Need to be more sophisticated if we want
            # to get sub-intervals (e.g. use tstart/tstop to match edges)
            #scale = np.average(scales,weights=edges[1:]-edges[:-1])
            #print scale
        elif (scale is not None):
            weights = scale*weights/(scale*weights+(1-weights))
            sexp = exp*(scale*self.S/self.E)
        else:
            sexp = exp*(self.S/self.E)

        # for now, just let background be scale free
        bexp = exp*(self.B/self.E)

        # now that exposure and event selection are done in topocentric,
        # if we're using barycentering, replace the bin edges with the
        # correct (uniform) barycentric times;
        if use_barycenter and (not ft1_is_bary):
            # replace starts/stops with barycenter times
            starts = bary_edges[:-1][exposure_mask]
            stops = bary_edges[1:][exposure_mask]

        if time_series_only:
            weights_vec = np.zeros(len(starts),dtype=float)
            weights2_vec = np.zeros(len(starts),dtype=float)
            idx = 0
            for i in range(len(starts)):
                if nweights[i] > 0:
                    w = weights[idx:idx+nweights[i]]
                    idx += nweights[i]
                    weights_vec[i] = np.sum(w)
                    weights2_vec[i] = np.sum(w**2)
            #minimum_exposure = minimum_exposure*(tcell/30)
            if use_barycenter:
                return CellTimeSeries(
                    starts,stops,exp,sexp,bexp,
                    nweights,weights_vec,weights2_vec,
                    alt_starts=topo_edges[:-1][exposure_mask],
                    alt_stops=topo_edges[1:][exposure_mask],
                    timesys='barycenter',minimum_exposure=0)
            else:
                return CellTimeSeries(
                    starts,stops,exp,sexp,bexp,
                    nweights,weights_vec,weights2_vec,
                    timesys='topocentric',minimum_exposure=0)
        # NB this hasn't been updated to properly incorporate the flexible
        # (time-dependent) scale!  So raise an error if that's used.
        if scale_series is not None:
            raise NotImplementedError("Scale series only supported for CellTimeSeries right now.")
        cells = deque()
        idx = 0
        SonB = self.S/self.B*(scale or 1)
        for i in range(len(starts)):
            t = times[idx:idx+nweights[i]]
            w = weights[idx:idx+nweights[i]]
            idx += nweights[i]
            c = Cell(starts[i],stops[i],sexp[i],t.copy(),w.copy(),SonB)
            cells.append(c)
        return list(cells)

    def get_cells_from_time_intervals(self,tstarts,tstops,
            randomize=False):
        """ Given a specific set of start and stop times, makes Cells.

        Ideally this would be merged with the more general method.
        """

        exp = self.get_exposure(tstops)-self.get_exposure(tstarts) 
        exp *= self.S/self.E
        start_idx = np.searchsorted(self.ti,tstarts)
        stop_idx = np.searchsorted(self.ti,tstops)
        if randomize:

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
            new_indices = np.searchsorted(cexp,np.random.rand(nphot))
            nweights = np.bincount(new_indices,minlength=ncell)

            # randomly permute the weights, and then draw random times
            # for each cell for the give number of photons in it
            a = np.argsort(new_indices)
            new_indices = new_indices[a]
            we = we[a]
            dts = (tstops-tstarts)[new_indices]
            ti = tstarts[new_indices] + np.random.rand(nphot)*dts

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
            randomize=False):
        """ Return Cells for all contiguous exposure cells between
        tstart and tstop.
        """
        starts,stops = self.get_contiguous_exposures(
                tstart=tstart,tstop=tstop)
        return self.get_cells_from_time_intervals(starts,stops,
                randomize=randomize)


class PhaseData(Data):
    """ Use phase instead of time, and ignore exposure.
    """

    def __init__(self,ft1files,weight_col,max_radius=None,
            pulse_phase_col='PULSE_PHASE',phase_shift=None,
            ra=None,dec=None):
        """ The FT1 files
            ra, dec of source (deg)
            weight_cut -- fraction of source photons to retain
            max_radius -- maximum photon separation from source [deg]
        """
        self.ft1files = ft1files
        self.ra = ra
        self.dec = dec
        if (max_radius is not None) and ((ra or dec) is None):
            raise ValueError('Must specify ra and dec for radius cut!')

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


    def get_cells(self,ncell=100,get_converse=False,randomize=False):
            
        edges = np.linspace(0,1,ncell+1)
        starts,stops = edges[:-1],edges[1:] 

        times = self.ti
        weights = self.we
        if get_converse:
            weights = 1-weights
        if randomize:
            times = np.random.rand(len(times))
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
        self.weights = np.empty(self.nph,dtype=np.float32)
        self.alpha_indices = np.empty(self.nph,dtype=int)
        ctr = 0
        for i,ic in enumerate(cells):
            nph = len(ic.we)
            s = slice(ctr,ctr+nph)
            self.weights[s] = ic.we
            self.alpha_indices[s] = i
            ctr += nph

        self._tmp1 = np.empty(self.nph,dtype=np.float32)
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
        no_zero_pad=False):
    """ Use FFT to evalute the sums in the maximum likelihood expression.

    This version matches the notation in the paper.

    tweak_exp -- make sure that the mean signal for source and background
        are 0; helps prevent spectral leakage from low frequencies

    get_amps -- if True, return frequencies, real amplitudes, imag. 
        amplitudes, and their uncertainties.  (NB these are the
        *fractional* modulation coefficients.)

    Returns: frequencies, P_0 (background fixed power spectrum), 
        P_1 (background-free spectrum), P_b (power spectrum of background)

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
        return freqs[:(l//4+1)],alpha_cos0,alpha_sin0,WW_cos,WW_sin

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
        a2 = np.mean(cph**2*ts.weights2)
        b2 = np.mean(sph**2*ts.weights2)
        rvals += 2*(a*cph + b*sph)
        pows[i] = (a**2/a2 + b**2/b2)

    return rvals,pows*ts.sexp.sum()**2/ts.sexp.shape[0]

def plot_clls_lc(rvals,ax=None,scale='linear',min_mjd=None,max_mjd=None,
        ul_color='C1',meas_color='C0'):
    """ Make a plot of the output lc CellsLogLikelihood.get_lightcurve.
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
    ul_mask = (rvals[:,-1] == -1) & (~np.isnan(rvals[:,-1]))
    t = rvals[ul_mask].transpose()
    ax.errorbar(t[0],t[2],xerr=t[1],yerr=0.2*(1 if scale=='linear' else t[2]),uplims=True,marker=None,color=ul_color,alpha=0.5,ls=' ',ms=3)
    t = rvals[~ul_mask].transpose()
    ax.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color=meas_color,alpha=0.5,ls=' ',ms=3)
    ax.set_xlabel('MJD')
    ax.set_ylabel('Relative Flux')

