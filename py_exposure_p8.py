"""
Module implements a relatively feature-complete computation of the LAT
exposure.  The purpose is primarily for fast computation of an exposure
time series for a very limited patch of sky.

This was written for Pass 7 data and has been roughly kluged to work with
Pass 8, at least commonly used event classes.

Requires: Fermi ScienceTools

author(s): Matthew Kerr
"""
from collections import deque
import os
from os.path import join

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d,splrep,BSpline

# fermitools
from . import pycaldb
from . import pypsf
from . import keyword_options
from .gti import Gti

dbug = dict()

DEG2RAD = np.pi/180.
EQUATORIAL = 1
GALACTIC = 0
DEFAULT_IRF ='P8R3_SOURCE_V3'

def get_contiguous_exposures(TSTART,TSTOP,tstart=None,tstop=None,
        max_interval=10,get_indices=False):
    """ Given a set of FT2-like entries with bounds TSTART/TSTOP, return the
    edges of the intervals for which the exposure is uninterrupted.

    This will typically be an orbit, or portions of an orbit if there is an
    SAA passage.

    Parameters
    ----------
    TSTART : starts of exposure intervals (MET)
    TSTOP : ends of exposure intervals (MET)
    tstart : optional start time for intervals to return (MET)
    tstop : optional end time for intervals to return (MET)
    max_interval -- maximum acceptable break in exposure for a
        contiguous interval [10s]
    get_indices : if True, return the index of the first entry in each new
        segment, viz. TSTART[idx]-TSTOP[idx-1] > max_interval
    """

    t0s = TSTART
    t1s = TSTOP
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
    # now assemble the complement
    good_starts = np.empty(len(break_starts)+1)
    good_stops = np.empty_like(good_starts)
    good_starts[0] = t0s[0]
    good_starts[1:] = break_stops
    good_stops[-1] = t1s[-1]
    good_stops[:-1] = break_starts
    if get_indices:
        return good_starts,good_stops,np.flatnonzero(break_mask)+1
    return good_starts,good_stops

def adjust_cosines(TSTART,TSTOP,pcosines,acosines=None,zcosines=None,
        oversample=False):
    """ Interpolate in polar angle, azimuth, and zenith.
    """

    dbug['T0'] = TSTART
    dbug['T1'] = TSTOP
    dbug['pcos0'] = pcosines
    dbug['acos0'] = acosines
    dbug['zcos0'] = zcosines
    assert(len(pcosines)==len(TSTART))
    _,_,idx = get_contiguous_exposures(TSTART,TSTOP,get_indices=True)
    edges = np.append(0,idx)
    if idx[-1] < len(pcosines):
        edges = np.append(edges,len(pcosines))

    N = len(pcosines)
    if oversample:
        N *= 2
    prvals = np.empty(N)
    arvals = np.empty(N) if (acosines is not None) else None
    zrvals = np.empty(N) if (zcosines is not None) else None
    # TMP
    prvals[:] = np.nan

    def interpolate(dx,y,oversample=False,clip_min=-1):

        if len(y) == 1:
            if oversample:
                return np.asarray([y[0],y[0]])
            return y

        # evaluate the slope and extrapolate the final point
        m = np.empty(len(y))
        m[:-1] = y[1:]-y[:-1]
        # in case the last bin is shorter than the preceding one, correct
        # the slop for its duration
        m[-1] = m[-2]*dx[-1]/dx[-2]

        if oversample:
            rvals = np.empty(len(y)*2)
            rvals[0::2] = y + m*0.25
            rvals[1::2] = y + m*0.75
        else:
            rvals = y + m*0.5

        return np.clip(rvals,clip_min,1,out=rvals)

    for i0,i1 in zip(edges[:-1],edges[1:]):
        dx = TSTOP[i0:i1] - TSTART[i0:i1]
        if oversample:
            mi0,mi1 = 2*i0,2*i1
        else:
            mi0,mi1 = i0,i1
        prvals[mi0:mi1] = interpolate(
                dx,pcosines[i0:i1],oversample=oversample)
        if arvals is not None:
            arvals[mi0:mi1] = interpolate(
                    dx,acosines[i0:i1],oversample=oversample,clip_min=0)
        if zrvals is not None:
            zrvals[mi0:mi1] = interpolate(
                    dx,zcosines[i0:i1],oversample=oversample)
    # TMP
    assert(not np.any(np.isnan(prvals)))
    dbug['pcos1'] = prvals
    dbug['acos1'] = arvals
    dbug['zcos1'] = zrvals
    return prvals,arvals,zrvals

def adjust_cosines2(TSTART,TSTOP,pcosines,acosines,zcosines,
        oversample=False):
    dbug['pcos0'] = pcosines.copy()
    dbug['acos0'] = acosines.copy()
    dbug['zcos0'] = zcosines.copy()
    dbug['t0'] = TSTART.copy()
    dbug['t1'] = TSTOP.copy()
    # find breaks in the orbit
    idx = np.flatnonzero(TSTART[1:]-TSTOP[:-1] > 30.1)
    i0s = np.append(0,idx+1)
    i1s = np.append(idx+1,len(TSTART)+1)
    if oversample:
        tmid = np.empty(len(TSTART)*2)
        dt = TSTOP-TSTART
        tmid[0::2] = TSTART + 0.25*dt
        tmid[1::2] = TSTART + 0.75*dt
    else:
        tmid = 0.5*(TSTART+TSTOP)

    n_in_seg = i1s-i0s
    mask = n_in_seg > 1

    N = 2*len(pcosines) if oversample else len(pcosines)
    pvals = np.full(N,np.nan)
    avals = np.full(N,np.nan)
    zvals = np.full(N,np.nan)

    if not oversample:
        carryover = i0s[~mask]
        pvals[carryover] = pcosines[carryover]
        avals[carryover] = acosines[carryover]
        zvals[carryover] = zcosines[carryover]
    else:
        for idx in i0s[~mask]:
            pvals[2*idx:2*(idx+1)] = pcosines[idx]
            avals[2*idx:2*(idx+1)] = acosines[idx]
            zvals[2*idx:2*(idx+1)] = zcosines[idx]

    i0s = i0s[mask]
    i1s = i1s[mask]

    if not oversample:
        dst_i0 = i0s
        dst_i1 = i1s
    else:
        dst_i0 = 2*i0s
        dst_i1 = 2*i1s

    for i0,i1,di0,di1 in zip(i0s,i1s,dst_i0,dst_i1):

        x0 = TSTART[i0:i1]
        xmid = tmid[di0:di1]

        s = splrep(x0,pcosines[i0:i1],k=1)
        pvals[di0:di1] = BSpline(*s,extrapolate=True)(xmid)

        s = splrep(x0,acosines[i0:i1],k=1)
        avals[di0:di1] = BSpline(*s,extrapolate=True)(xmid)

        s = splrep(x0,zcosines[i0:i1],k=1)
        zvals[di0:di1] = BSpline(*s,extrapolate=True)(xmid)

    print((avals<0).sum())
    print((avals>1).sum())
    print((pvals>1).sum())
    assert(not np.any(np.isnan(pvals)))
    assert(not np.any(np.isnan(avals)))
    assert(not np.any(np.isnan(zvals)))
    avals[avals<0] = np.abs(avals[avals<0])
    avals[avals>1] = 2-(avals[avals>1])
    pvals[pvals>1] = 2-(pvals[pvals>1])
    zvals[zvals>1] = 2-(zvals[zvals>1])
    dbug['pcos1'] = pvals.copy()
    dbug['acos1'] = avals.copy()
    dbug['zcos1'] = zvals.copy()
    return pvals,avals,zvals

# This is an adhoc attempt to correct exposure to bin center AFTER collapse
# to a scalar.  Fast but not very useful...?
def adjust_exposure(TSTART,TSTOP,exposure):

    assert(len(exposure)==len(TSTART))
    _,_,idx = get_contiguous_exposures(TSTART,TSTOP,get_indices=True)
    edges = np.append(0,idx)
    if idx[-1] < len(exposure):
        edges = np.append(edges,len(exposure))

    rvals = np.empty_like(exposure)
    # TMP
    rvals[:] = np.nan

    def interpolate(x,y,xmid):

        if len(x) == 1:
            return y
        elif len(x) == 2:
            kind = 'linear'
        else:
            kind = 'quadratic'

        ip = interp1d(x,y,kind=kind,bounds_error=False,
                fill_value='extrapolate')
        return np.maximum(0,ip(xmid))

    for i0,i1 in zip(edges[:-1],edges[1:]):
        x0 = TSTART[i0:i1]
        xmid = 0.5*(x0+TSTOP[i0:i1])
        rvals[i0:i1] = interpolate(x0,exposure[i0:i1],xmid)
    # TMP
    assert(not np.any(np.isnan(rvals)))
    return rvals

class InterpTable(object):
    """ Implement 2d bilinear or nearest neighbor interpolation.

    Bilinear will automatically extrapolate past bin centers.

    This is largely designed for interpolation in log10(E) and cos(theta).
    """

    def __init__(self,x,y,z):

        x0 = x[0]-(x[1]-x[0])
        x1 = x[-1]+(x[-1]-x[-2])
        x = np.concatenate(([x0],x,[x1]))
        y0 = y[0]-(y[1]-y[0])
        y1 = y[-1]+(y[-1]-y[-2])
        y = np.concatenate(([y0],y,[y1]))
        z = self._augment_data(z)

        self._x = x
        self._y = y
        self._z = z
        self._xmid = 0.5*(self._x[1:]+self._x[:-1])
        self._ymid = 0.5*(self._y[1:]+self._y[:-1])
        self._dx = x[1]-x[0]
        self._dy = y[1]-y[0]

    def _augment_data(self,data):
        """ Build a copy of data extrapolated by one sample."""

        d = np.empty([data.shape[0]+2,data.shape[1]+2])

        # copy original data into interior
        d[1:-1,1:-1] = data

        # add bottom row (not corners)
        d[0,1:-1] = data[0] - (data[1]-data[0])

        # add top row (not corners)
        d[-1,1:-1] = data[-1] + (data[-1]-data[-2])

        # add left side (not corners)
        d[1:-1,0] = data[:,0] - (data[:,1]-data[:,0])

        # add right side (not corners)
        d[1:-1,-1] = data[:,-1] + (data[:,-1]-data[:,-2])

        # corners
        d[0,0] = d[1,0] + d[0,1] - data[0,0]
        d[0,-1] = d[1,-1] + d[0,-2] - data[0,-1]
        d[-1,0] = d[-2,0] + d[-1,1] - data[-1,0]
        d[-1,-1] = d[-2,-1] + d[-1,-2] - data[-1,-1]

        return d

    def __call__(self,x,y,bilinear=True):

        if not bilinear:
            i = np.searchsorted(self._xmid,x)
            j = np.searchsorted(self._ymid,y)
            return self._z[i,j]

        i = np.searchsorted(self._x,x)-1
        j = np.searchsorted(self._y,y)-1

        x2,x1 = self._x[i+1],self._x[i]
        y2,y1 = self._y[j+1],self._y[j]
        f00 = self._z[i,j]
        f11 = self._z[i+1,j+1]
        f10 = self._z[i+1,j]
        f01 = self._z[i,j+1]
        norm = 1./((x2-x1)*(y2-y1))
        return ( (x2-x)*(f00*(y2-y)+f01*(y-y1)) + (x-x1)*(f10*(y2-y)+f11*(y-y1)) )*norm

class Binning(object):
    """ Specify the binning for a livetime calculation."""

    defaults = (
        ('theta_bins',np.linspace(0.4,1,21),'bins in cosine(incidence angle)'),
        ('phi_bins',None,'bins in azimuth (radians); default is a single bin'),
        ('time_bins',None,'bins in MET; default is a single bin')
    )

    @keyword_options.decorate(defaults)
    def __init__(self,**kwargs):
        keyword_options.process(self,kwargs)
        if self.phi_bins is not None:
            if (self.phi_bins[0] < 0) or (self.phi_bins[-1] > np.pi/2):
                print('Warning, azimuth angles are wrapped to 0 to pi/2')

    def equals(self,bins):
        equal = True
        for key in ['theta_bins','phi_bins','time_bins']:
            b1 = self.__dict__[key]; b2 = bins.__dict__[key]
            if b1 is not None:
                test = (b2 is not None) and np.allclose(b1,b2)
            else: test = b2 is None
            equal = equal and test
        return equal

class Livetime(object):
    """Calculate the livetime as a function of incidence angle using the GTI
       specified in a collection of FT1 files and the livetime entries from
       an FT2 file.

       The default implementation uses the native resolution of the FT2
       file, i.e., when the user requests the livetime, the values for the
       S/C z-axis and zenith positions are used directly to calculate the
       incidence/zenith angles.

       This executes with comparable speed (factor of ~2 slower) to the
       Science Tools application gtltcube (which bins position)."""

    defaults = (
        ('verbose',1,'verbosity level'),
        ('gti_mask',None,'additional excisions'),
        ('tstart',0,'lower time limit in MET'),
        ('tstop',1e100,'upper time limit in MET'),
        ('remove_zeros',True,'remove bins with 0 livetime'),
        ('fast_ft2',True,'use fast algorithm for calculating GTI overlaps'),
        ('override_ltfrac',None,'use this livetime fraction instead of FT2 vals'),
        ('deadtime',False,'accumulate deadtime instead of livetime')
    )

    @keyword_options.decorate(defaults)
    def __init__(self,ft2files,ft1files,**kwargs):
        keyword_options.process(self,kwargs)
        self.prev_vals = self.prev_ra = self.prev_dec = None # initialize caching
        self.fields    = ['START','STOP','LIVETIME','RA_SCZ','DEC_SCZ',
                            'RA_ZENITH','DEC_ZENITH','RA_SCX','DEC_SCX']
        self._setup_gti(ft1files)
        self._setup_ft2(ft2files)
        self._update_gti()
        self._process_ft2()
        self._finish()

    def _finish(self):
        for field in self.fields:
            if ('DEC_' in field):
                self.__dict__['COS_'+field] = np.cos(self.__dict__[field])
                self.__dict__['SIN_'+field] = np.sin(self.__dict__[field])
        if self.deadtime:
            self.LIVETIME = self.LIVETIME*((1-self.LTFRAC)/self.LTFRAC)
        if self.override_ltfrac is not None:
            self.LIVETIME *= self.override_ltfrac/self.LTFRAC
            self.LTFRAC    = self.override_ltfrac

    def _setup_gti(self,ft1files):
        """ Take the union of all GTIs provided by FT1 files, then take an 
            intersection with the (optional) gti_mask and the time limits.
        """
        if self.verbose >= 2: print('Processing GTI...')
        if not hasattr(ft1files,'__iter__'): ft1files = [ft1files]
        gti = self.gti = Gti(ft1files[0])
        if len(ft1files) > 1:
            for ft1 in ft1files[1:]: gti.combine(Gti(ft1))
        tmin = float(max(gti.minValue(),self.tstart or 0))
        tmax = float(min(gti.maxValue(),self.tstop or 1e100))
        gti = self.gti = gti.applyTimeRangeCut(tmin,tmax)
        if self.gti_mask is not None:
            before = round(gti.computeOntime())
            gti.intersection(self.gti_mask)
            if verbose >= 1:
                print('Applied GTI mask; ontime reduced from %ds to %ds'%(
                        before,round(gti.computerOntime())))

        self.gti_starts = np.sort(gti.get_edges(True))
        self.gti_stops = np.sort(gti.get_edges(False))
        if self.verbose >= 1:
            print('Finished computing GTI from FT1 files; total ontime = %ds'%(
                    round(gti.computeOntime())))

    def _update_gti(self):
        """ Trim GTI to FT2 boundaries.
        This is a sanity check, essentially, on GTI, to make sure they
        are consistent with the contents of the FT2 file.  This should
        normally be the case, but if the FT2 file is missing data, the GTI
        should be updated to avoid including data without S/C pointing
        history.

        Subsequently, event data should be masked according to the times.
        """
        # The easiest way to do this is to identify gaps in the FT2 file.
        # These are typically *exactly* the same as the GTI already defined
        # in the FT2 file, with two exceptions: when the FT2 file doesn't
        # extend to the same length as the FT1 file(s), or when the FT2
        # file has missing coverage.  The former is benign, while the
        # latter indicates a problem with the input files.

        # May 15, 2019 -- TODO; this code is on the right track, but to
        # cover all cases, need to consider the case that a GTI could have
        # multiple gaps within it.  (What's coded below allowed for multi
        # GTI per gap.)  So leave the sanity check, but don't actually
        # change any GTI for now.
        
        gap_idx = np.ravel(np.argwhere((self.START[1:]-self.STOP[:-1]) > 0))
        gap_starts = self.STOP[gap_idx]
        gap_stops = self.START[gap_idx+1]

        g0 = self.gti_starts
        g1 = self.gti_stops

        i0 = np.searchsorted(gap_stops-1e-6,g0)
        np.clip(i0,0,len(gap_stops)-1,out=i0)
        starts_in_gap = (g0 > gap_starts[i0]) & (i0 < len(gap_stops)-1)

        i1 = np.searchsorted(gap_stops+1e-6,g1)
        np.clip(i1,0,len(gap_stops)-1,out=i1)
        ends_in_gap = (g1 > gap_starts[i1]) & (i1 < len(gap_stops)-1)

        if np.any(starts_in_gap | ends_in_gap):
            print(""""WARNING!!!
            
            GTI are present with no spacecraft pointing information!  
            This likely represents a problem with the input file. 
            This algorithm will attempt to compute the exposure correctly
            by adjusting the GTI, but you will need to apply the resulting
            mask to the events directly.
            """)

            print(g0[starts_in_gap])

        # adjust all GTI to gap boundaries
        g0[starts_in_gap] = gap_stops[i0[starts_in_gap]]
        g1[ends_in_gap] = gap_starts[i1[ends_in_gap]]

        # finally, adjust all GTI that precede or follow the FT2 file
        g0[g1 < self.START[0]] = self.START[0]
        g1[g0 > self.STOP[-1]] = self.STOP[-1]

        # remove any GTI that now have negative length
        mask = (g1-g0) > 0
        self.gti_starts = g0[mask]
        self.gti_stops = g1[mask]

    def _setup_ft2(self,ft2files):
        """Load in the FT2 data.  Optionally, mask out values that will not
           contibute to the exposure."""
        if self.verbose >= 2: print('Loading FT2 files...')
        if not hasattr(ft2files,'__iter__'): ft2files = [ft2files]
        load_data = dict()
        for field in self.fields:
            load_data[field] = deque()
        for ift2file,ft2file in enumerate(ft2files):
            if self.verbose > 1:
                print('...Loading FT2 file # %d'%(ift2file))
            with fits.open(ft2file,memmap=False) as handle:
                for field in self.fields:
                    load_data[field].append(
                            handle['SC_DATA'].data.field(field))
        for field in self.fields:
            self.__dict__[field] = np.concatenate(load_data[field])
            if ('RA_' in field) or ('DEC_' in field):
                self.__dict__[field] *= DEG2RAD
        # trim GTI to FT2 range
        if (self.gti_starts[0] < self.START[0]) or (self.gti_stops[-1] > self.STOP[-1]):
            self.gti = self.gti.applyTimeRangeCut(self.START[0],self.STOP[-1])
            self.gti_starts = np.sort(self.gti.get_edges(True))
            self.gti_stops = np.sort(self.gti.get_edges(False))
            mask = (self.STOP > self.gti_starts[0]) | (self.START < self.gti_stops[-1])
            self.mask_entries(mask)
        if self.remove_zeros:
            mask = self.LIVETIME > 0
            self.mask_entries(mask)

        # ensure entries are sorted by time
        self.mask_entries(np.argsort(self.START)) 
        if self.verbose > 1: print('Finished loading FT2 files!')
  
    def _process_ft2(self):
        if self.verbose >= 2: print('Processing the FT2 file (calculating overlap with GTI)...')
        func = self._process_ft2_fast if self.fast_ft2 else \
               self._process_ft2_slow
        overlaps = func(self.gti_starts,self.gti_stops)
        # sanity check code on overlaps
        if np.any(overlaps > 1):
            print('Error in overlap calculation!  Results may be unreliable.')
            overlaps[overlaps>1] = 1
        # TEMPORARY
        self._overlaps = overlaps.copy()
        # END TEMPORARY
        mask = overlaps > 0
        if self.remove_zeros:
            self.mask_entries(mask)
            overlaps = overlaps[mask]
        self.LTFRAC = self.LIVETIME/(self.STOP-self.START)
        self.fields += ['LTFRAC']
        self.LIVETIME *= overlaps
        if self.verbose > 1: print('Finished processing the FT2 file!')

    def _process_ft2_slow(self,gti_starts,gti_stops):
        """Calculate the fraction of each FT2 interval lying within the GTI.
           Uses a slow, easily-checked algorithm.
           The complexity is O(t^2) and is prohibitive
           for mission-length files."""
        t1,t2,lt = self.START,self.STOP,self.LIVETIME
        overlaps = np.zeros_like(lt)
        for i,(gti_t1,gti_t2) in enumerate(zip(gti_starts,gti_stops)):
            maxi = np.maximum(gti_t1,t1)
            mini = np.minimum(gti_t2,t2)
            overlaps += np.maximum(0,mini - maxi)
        return overlaps/(t2 - t1)
        
    def _process_ft2_fast(self,gti_starts,gti_stops):
        """Calculate the fraction of each FT2 interval lying within the GTI.
           Use binary search to quickly process the FT2 file.
           The complexity is O(t*log(t))."""
        # Development notes: one bug results from GTIs that reside partially
        # or completely outside of regions of validity in the FT2.  I have
        # implemented a fix that trims GTI ranges to the edges of the FT2
        # boundaries when this arises, but I am not 100% sure this is
        # correct.  I have also implemented some sanity checks elsewhere
        # in the code, but at some point it would be good to verify this
        # implementation.  May 15, 2019
        t1,t2,lt = self.START,self.STOP,self.LIVETIME
        gti_t1,gti_t2 = gti_starts,gti_stops
        overlaps = np.zeros_like(lt)
        i1 = np.searchsorted(t2,gti_t1) # NB. -- t2 in both not a typo
        i2 = np.searchsorted(t2,gti_t2)
        seps = i2 - i1
        for i,(ii1,ii2) in enumerate(zip(i1,i2)):
            overlaps[ii1:ii2+1] = 1. # fully-contained FT2 intervals
            if seps[i] > 0: # correct the endpoint FT2 intervals
                overlaps[ii1] = (t2[ii1] - gti_t1[i])/(t2[ii1] - t1[ii1])
                overlaps[ii2] = (gti_t2[i] - t1[ii2])/(t2[ii2] - t1[ii2])
                # May 15, 2019; above implementation can lead to incorrect
                # results if the GTI lies outside of the range of validity
                # of an FT2 file, which can now apparently happen.  Trimming
                # the GTI to the FT2 validity is equivalent, in this case,
                # to capping the overlaps at 1, so simply do that as an
                # array operation at the end of the computation.
            else: # edge case with exceptionally short GTI
                a = max(t1[ii1],gti_t1[i])
                b = min(t2[ii1],gti_t2[i])
                overlaps[ii1] = max(b-a,0)/(t2[ii1] - t1[ii1])
        # May 15, 2019 -- below array operation MAY account for GTIs lying
        # outside of FT2 START/STOP ranges
        np.clip(overlaps,0,1,out=overlaps)
        return overlaps

    def mask_entries(self,mask=None):
        """If make is None, assume a LIVETIME > 0 cut."""
        if mask is None:
            mask = self.LIVETIME > 0
        for field in self.fields:
            self.__dict__[field] = self.__dict__[field][mask]

    def get_cosines(self,ra,dec,theta_cut,zenith_cut,get_phi=False,
            apply_correction=False,oversample=False):
        """ Return the cosine of the arclength between the specified 
            direction and the S/C z-axis and [optionally] the azimuthal
            orientation as a cosine.

        Parameters
        ----------
        ra : right ascension (radians)
        dec : declination (radians)
        theta_cut : cosine(theta_max)
        zenith_cut : cosine(zenith_max)
        get_phi : return the polar cosines too
        apply_correction : if True, correct the S/C orientation to the
            center of the FT2 interval.  The data are tabulated at START,
            but the livetime is accumulated over START to STOP.  The
            correction is an interpolation between neighboring entries.
            This can be a bit slow.
        oversample: replace every FT2 bin by one half its size with the
            interpolated S/C attitude

        Returns:
        mask -- True if the FT2 interval satisfies the specified cuts
        pcosines -- cosines of polar angles
        acosines -- cosines of azimuthal angles [optional]
        """    
        # all of these calculations basically determine the dot product
        # between the source position and the vector of interest, viz. the
        # S/C Z- and X-axes, and the zenith direction.
        ra_s,ra_z = self.RA_SCZ,self.RA_ZENITH
        cdec,sdec = np.cos(dec),np.sin(dec)

        # cosine(polar angle) of source in S/C system
        pcosines  = self.COS_DEC_SCZ*cdec*np.cos(ra-self.RA_SCZ) + self.SIN_DEC_SCZ*sdec

        # cosine(polar angle) between source and zenith
        zcosines = self.COS_DEC_ZENITH*cdec*np.cos(ra-self.RA_ZENITH) + self.SIN_DEC_ZENITH*sdec

        # make a preliminary rough cut to speed things up
        mask = (pcosines > (theta_cut-0.15)) & (zcosines >= (zenith_cut-0.15))
        pcosines = pcosines[mask]
        zcosines = zcosines[mask]
        T0 = self.START[mask]
        T1 = self.STOP[mask]
        LT = self.LIVETIME[mask]

        if get_phi:
            ra_s = self.RA_SCX
            acosines = self.COS_DEC_SCX[mask]*cdec*np.cos(ra-self.RA_SCX[mask]) + self.SIN_DEC_SCX[mask]*sdec
            # the (1-pcosine^2)**0.5 normalizes the vector to the X/Y plane,
            # so that the remaining portion is just the source vector
            # projected onto the x axis; and the abs/clip folds to 0 to pi/2
            # to enforce the four-fold symmetry
            np.clip(np.abs(acosines/(1-pcosines**2)**0.5),0,1,out=acosines)
        else:
            acosines = None

        if oversample:
            pcosines,acosines,zcosines = adjust_cosines2(
                    T0,T1,pcosines,acosines,zcosines,oversample=True)
            dT = T1-T0
            new_T0 = np.empty(len(T0)*2)
            new_T0[0::2] = T0
            new_T0[1::2] = T0 + 0.5*dT
            new_T1 = np.empty(len(T0)*2)
            new_T1[0::2] = new_T0[1::2]
            new_T1[1::2] = T1
            new_LT = np.empty(len(T0)*2)
            new_LT[0::2] = 0.5*self.LIVETIME[mask]
            new_LT[1::2] = 0.5*self.LIVETIME[mask]
            new_mask = np.empty(len(mask)*2,dtype=bool)
            new_mask[0::2] = mask
            new_mask[1::2] = mask
            assert(len(pcosines)==len(new_T0))
            T0 = new_T0
            T1 = new_T1
            LT = new_LT
            mask = new_mask

        elif apply_correction:
            pcosines,acosines,zcosines = adjust_cosines2(
                    T0,T1,pcosines,acosines,zcosines,oversample=False)

        # make final mask
        fmask = (pcosines >= theta_cut) & (zcosines >= zenith_cut)
        mask[mask] = fmask
        if acosines is not None:
            acosines = acosines[fmask]
        return mask,pcosines[fmask],acosines,T0[fmask],T1[fmask],LT[fmask]

    def _do_bin(self,binning,mask,pcosines,acosines,time_range=None):
        weights = self.LIVETIME[mask]
        if time_range is not None:
            weights = weights * self._process_ft2_fast([time_range[0]],[time_range[1]])[mask]
        if binning.phi_bins is None:    
            return np.histogram(pcosines,bins=binning.theta_bins,weights=weights)
        else:
            return np.histogram2d(pcosines,acosines,bins=[binning.theta_bins,binning.phi_bins],weights=weights)

    def __call__(self,skydir,binning,theta_cut=0.4,zenith_cut=-1):
        """ Return the exposure at location indicated by skydir using the
            theta/time/phi binning specified by the Binning object.
            
            binning can also be passed as an integer, in which case it
            is assumed to specify the number of polar bins from
            theta_cut to 1, and no binning in time or azimuth is performed."""
      
        ra,dec = DEG2RAD*skydir.ra(),DEG2RAD*skydir.dec()
        if np.isscalar(binning):
            binning = Binning(theta_bins=np.linspace(theta_cut,1,binning+1))
        if (ra==self.prev_ra) and (dec==self.prev_dec) and binning.equals(self.prev_binning):
            return self.prev_val
        else: self.prev_ra,self.prev_dec,self.prev_binning = ra,dec,binning
        mask,pcosines,acosines = self.get_cosines(ra,dec,theta_cut,zenith_cut,binning.phi_bins is not None)

        if binning.time_bins is None:
            self.prev_val = self._do_bin(binning,mask,pcosines,acosines)
        else:
            self.prev_val = [self._do_bin(binning,mask,pcosines,acosines,time_range=[t0,t1])
                for t0,t1 in zip(binning.time_bins[:-1],binning.time_bins[1:])]
        return self.prev_val

    def get_gti_mask(self,timestamps):
        """ Return a mask with the same shape as timestamps according to
            whether the timestamps lie within a GTI.
        """
        event_idx = np.searchsorted(self.gti_stops,timestamps)
        mask = event_idx < len(self.gti_stops)
        mask[mask] = timestamps > self.gti_starts[event_idx[mask]]
        if self.verbose >= 1:
            print('gti mask: %d/%d'%(mask.sum(),len(mask)))
        return mask

class BinnedLivetime(Livetime):
    """See remarks for Livetime class for general information.
       
       This class provides an implementation of the livetime calculation
       in which the FT2 entries for the S/Z z-axis and zenith positions
       are binned onto a Healpix grid, allowing for a faster calculation
       with long FT2 files.
    """

    def finish(self):
        hp = Healpix(self.nside,Healpix.RING,EQUATORIAL)
        ras,decs = np.asarray( [hp.py_pix2ang(i) for i in range(12*self.nside**2)]).transpose()
        self.COS_HP_DEC = np.cos(decs)
        self.SIN_HP_DEC = np.sin(decs)
        self.HP_RA = ras
        ra_s,dec_s = self.RA_SCZ,self.DEC_SCZ
        ra_z,dec_z = self.RA_ZENITH,self.DEC_ZENITH
        self.S_PIX = np.fromiter((hp.py_ang2pix(ra_s[i],dec_s[i]) for i in range(len(ra_s))),dtype=int)
        self.Z_PIX = np.fromiter((hp.py_ang2pix(ra_z[i],dec_z[i]) for i in range(len(ra_z))),dtype=int)

    def __init__(self,nside=59,*args,**kwargs):
        raise NotImplementedError('This needs a replacement for Healpix.')
        self.nside = nside
        super(BinnedLivetime,self).__init__(*args,**kwargs)

    def get_cosines(self,skydir):
        ra,dec    = np.radians([skydir.ra(),skydir.dec()])

        # calculate the arclengths to the various Healpix
        cosines  = self.COS_HP_DEC*np.cos(dec)*np.cos(ra-self.HP_RA) + self.SIN_HP_DEC*np.sin(dec)
        scosines = cosines[self.S_PIX]
        if self.zenithcut > -1:
            zcosines = cosines[self.Z_PIX]
            mask = (scosines>=self.fovcut) & (zcosines>=self.zenithcut)
        else:
            mask = (scosines>=self.fovcut)
        return scosines,mask

class EfficiencyCorrection(object):
    """ Apply a trigger-rate dependent correction to the livetime."""
    
    def __init__(self,irf=DEFAULT_IRF):
        self.irf = irf
        cdbm = pycaldb.CALDBManager(self.irf)
        hdus = [fits.open(f) for f in cdbm.get_aeff()]
        self._p0s = dict()
        self._p1s = dict()
        for hdu in hdus:
            for table in hdu:
                dat = table.data

                if table.name.startswith('EFFICIENCY_PARAMS'):
                    event_type = table.name.split('_')[-1]
                    self._p0s[event_type] = dat['efficiency_pars'][0]
                    self._p1s[event_type] = dat['efficiency_pars'][1]
            hdu.close()

    def _p(self,logE,v):
        a0,b0,a1,logEb1,a2,logEb2 = v
        b1 = (a0 - a1)*logEb1 + b0
        b2 = (a1 - a2)*logEb2 + b1
        if logE < logEb1:
            return a0*logE + b0
        if logE < logEb2:
            return a1*logE + b1
        return a2*logE + b2

    def __call__(self,e,ltfrac,event_type):
        """
        Return the trigger efficiency as estimate from the livetime frac.

        Parameters
        ----------
        e : energy (MeV)
        ltfrac : livetime fraction [0-1]
        event_type : e.g. FRONT, BACK, PSF0, ..., PSF3
        """
        loge = np.log10(e)
        p0 = self._p(loge,self._p0s[event_type])
        p1 = self._p(loge,self._p1s[event_type])
        return p0*ltfrac + p1

class EffectiveArea(object):

    def __init__(self,irf=DEFAULT_IRF,CALDB=None,use_phidep=False):
        """ Encapsulate reading the Fermi-LAT effective area.

        Parameters
        ----------
        irf : [DEFAULT_IRF] IRF to use
        CALDB : [None] path to override environment variable
        use_phidep : [False] use azmithual dependence for effective area
        """
        self.irf = irf
        cdbm = pycaldb.CALDBManager(self.irf)
        hdus = [fits.open(f) for f in cdbm.get_aeff()]
        self._aetabs = dict()
        self._aeffs = dict()
        self._p0tabs = dict()
        self._p1tabs = dict()
        for hdu in hdus:
            for table in hdu:
                dat = table.data
                try:
                    if not 'ENERG_LO' in table.columns.names:
                        continue
                except AttributeError:
                    continue
                elo = np.squeeze(dat['energ_lo'])
                ehi = np.squeeze(dat['energ_hi'])
                ecens = 0.5*(np.log10(elo) + np.log10(ehi))
                clo = np.squeeze(dat['ctheta_lo'])
                chi = np.squeeze(dat['ctheta_hi'])
                ccens = 0.5*(clo + chi)
                nc = len(clo)
                ne = len(elo)

                if table.name.startswith('EFFECTIVE AREA'):
                    event_type = table.name.split('_')[-1]
                    ae = np.reshape(dat['effarea'],(nc,ne))
                    # convert to cm^2 and swap to energy/cos(theta) order
                    ae = ae.transpose()*1e4
                    self._aeffs[event_type] = [elo,ehi,clo,chi,ae]
                    self._aetabs[event_type] = InterpTable(ecens,ccens,ae)

                elif table.name.startswith('PHI_DEPENDENCE'):
                    event_type = table.name.split('_')[-1]
                    p0 = np.reshape(dat['phidep0'],(nc,ne)).T
                    p1 = np.reshape(dat['phidep1'],(nc,ne)).T
                    self._p0tabs[event_type] = InterpTable(ecens,ccens,p0)
                    self._p1tabs[event_type] = InterpTable(ecens,ccens,p1)
            hdu.close()

    def _phi_mod(self,e,c,phi,event_type):
        # assume phi has already been reduced to range 0 to pi/2
        if phi is None:
            return 1.
        par0 = self._p0tabs[event_type](e,c,bilinear=False)
        par1 = self._p1tabs[event_type](e,c,bilinear=False)
        norm = 1. + par0/(1. + par1)
        phi = 2*abs((2./np.pi)*phi - 0.5)
        return (1. + par0*phi**par1)/norm

    def __call__(self,e,ctheta,event_type,phi=None,bilinear=True):
        """ Return bilinear (or nearest-neighbour) interpolation.

        Parameters
        ----------
        e : energy (MeV)
        ctheta : cosine incidence angle
        event_type : e.g. FRONT, BACK, PSF0, ..., PSF3
        phi : azimuthal angle mod pi/2 (rad)
        bilinear : use bilinear interpolation for effective area calc

        e and ctheta must be the same shape, or one must be a scalar.
        """
        e = np.log10(np.atleast_1d(e))
        c = np.atleast_1d(ctheta)
        if (len(e) > 1) and (len(c) > 1):
            assert(len(e)==len(c))
        aeff = self._aetabs[event_type](e,c,bilinear=bilinear)
        return aeff * self._phi_mod(e,c,phi,event_type)

    def image(self,event_types=['FRONT','BACK'],logea=False,fig_base=2,ctheta=0.99,show_image=False):

        effarea = np.sum([self._aeffs[et][-1] for et in event_types],axis=0)
        elo,ehi,clo,chi,_ = self._aeffs[event_types[0]]
        ebins = np.append(elo,ehi[-1])
        cbins = np.append(clo,chi[-1])

        import pylab as pl

        if show_image:
            #Generate a pseudo-color plot of the full effective area
            pl.figure(fig_base);pl.clf()
            pl.gca().set_xscale('log')
            if logea:
                pl.gca().set_yscale('log')
            pl.pcolor((ebins[:-1]*ebins[1:])**0.5,(cbins[:-1]+cbins[1:])/2.,effarea.transpose(),shading='auto')
            pl.title('Effective Area')
            pl.xlabel('$\mathrm{Energy\ (MeV)}$')
            pl.ylabel(r'$\mathrm{cos(\theta)}$')
            cb = pl.colorbar()
            cb.set_label('$\mathrm{Effective\ Area\ (m^2)}$')

        colors = ['blue','red','orange','green']
        pl.figure(fig_base+2)
        pl.clf()
        pl.gca().set_xscale('log')
        if logea:
            pl.gca().set_yscale('log')

        energies = np.logspace(np.log10(ebins[0]),np.log10(ebins[-1]),32*(len(ebins)-1)+1)
        for iet,et in enumerate(event_types):
            vals = np.squeeze([self(e,ctheta,et,bilinear=True) for e in energies]).transpose()
            pl.plot(energies,vals,label=f'{et} bilinear interp.',color=colors[iet])
            vals = np.squeeze([self(e,ctheta,et,bilinear=False) for e in energies]).transpose()
            pl.plot(energies,vals,label=f'{et} nearest-neighbour interp.',color=colors[iet])
        pl.title('On-axis Effective Area')
        pl.xlabel('$\mathrm{Energy\ (MeV)}$')
        pl.ylabel('$\mathrm{Effective\ Area\ (cm^2)}$')
        pl.legend(loc = 'lower right')
        pl.grid()

class PSFCorrection(object):
    """ Provide the PSF containment fraction at various incidence angles.
    """

    def __init__(self,irf=DEFAULT_IRF,radius=3):
        """
        Parameters
        ----------
        radius : aperture radius in degrees
        """
        self._psf = pypsf.CALDBPsf(irf=irf)
        self._rad = np.radians(radius)
        self._tables = dict()
        for event_type in self._psf.event_types():
            ecens = self._psf.ecens(event_type)
            ccens = self._psf.ccens(event_type)
            vals = np.empty((len(ecens),len(ccens)))
            for ie,e in enumerate(ecens):
                vals[ie] = self._psf.integral(e,event_type,self._rad)
            table = InterpTable(np.log10(ecens),ccens,vals)
            self._tables[event_type] = table

    def __call__(self,e,ctheta,event_type):
        """ Return the PSF containment for energy and incidence angle.

        Parameters
        ----------
        e : energy (MeV)
        ctheta : cosine incidence angle
        event_type : e.g. FRONT, BACK, PSF0, ..., PSF3
        """
        return self._tables[event_type](np.log10(e),ctheta)

def test_effective_area():

    ea = EffectiveArea(irf=DEFAULT_IRF)
    X,Y = np.meshgrid(np.arange(ea.ecens.shape[0]),np.arange(ea.ccens.shape[0]))
    ecens = (10**ea.ecens)[X.transpose()]
    ccens = ea.ccens[Y.transpose()]

    for bi in [True,False]:
        faeff,baeff = ea(ecens,ccens,bilinear=bi)
        #return faeff,ea.feffarea
        assert(np.allclose(faeff,ea.feffarea))
        assert(np.allclose(baeff,ea.beffarea))
        faeff = ea(ecens,ccens,bilinear=bi,event_class=0)
        baeff = ea(ecens,ccens,bilinear=bi,event_class=1)
        assert(np.allclose(faeff,ea.feffarea))
        assert(np.allclose(baeff,ea.beffarea))

    # check that we extrapolate up to cos(theta) = 1 reliably
    # TODO

    return



    import pylab as pl
    ea = EffectiveArea(use_phidep=True)
    edom = np.logspace(2,5,5)
    cdom = np.linspace(0.4,1.0,101)
    aeff = np.empty((len(edom),len(cdom)))
    phi = np.linspace(0,np.pi*0.5,101)
    for ie in range(len(edom)):
        aeff[ie] = ea(edom[ie],cdom,phi=phi)
    #return aeff
    pl.clf()
    pl.imshow(aeff,interpolation='nearest',aspect='auto')
    return ea,aeff

def test_psf_correction():
    pc = PSFCorrection(irf='P8R3_SOURCE_V3')

    # checks on PSF itself: FRONT/BACk
    cfac = pc._psf.integral(100,'FRONT',np.radians(3))
    # cos theta bins increase, so these should be increasing too
    assert(np.all(np.sort(cfac)==cfac))
    # rough sanity checks on integral magnitude
    cfac = pc._psf.integral(1000,'FRONT',np.radians(3))
    assert(np.all(cfac>0.9))
    cfac = pc._psf.integral(1000,'FRONT',np.radians(10))
    assert(np.all(cfac>0.99))
    # check back/front sense
    cfac = pc._psf.integral(1000,'FRONT',np.radians(3))
    assert(np.all(cfac>pc._psf.integral(1000,'BACK',np.radians(3))))

    # checks on PSF itself: PSF Types
    cfac = pc._psf.integral(100,'PSF3',np.radians(3))
    # cos theta bins increase, so these should be increasing too
    assert(np.all(np.sort(cfac)==cfac))
    # rough sanity checks on integral magnitude
    cfac = pc._psf.integral(1000,'PSF3',np.radians(3))
    assert(np.all(cfac[1:]>0.9)) # PSF3 very bad at cos(theta)~0.4...
    cfac = pc._psf.integral(1000,'PSF3',np.radians(10))
    assert(np.all(cfac>0.95))
    assert(np.all(cfac[2:]>0.99))
    #cfac = pc._psf.integral(1000,'PSF3',np.radians(3))
    #assert(np.all(cfac>pc._psf.integral(1000,'PSF0',np.radians(3))))

    # checks on the table
    ccens = pc._psf.ccens('FRONT')
    eeval = 10**2.125
    ffac = pc._psf.integral(eeval,'FRONT',np.radians(3))
    bfac = pc._psf.integral(eeval,'BACK',np.radians(3))
    ftest = pc(eeval,'FRONT',ccens)
    btest = pc(eeval,'BACK',ccens)
    assert(np.allclose(ftest,ffac))
    assert(np.allclose(btest,bfac))
    ftest = pc(100,'FRONT',ccens)
    btest = pc(100,'BACK',ccens)
    assert(np.all(ftest < ffac))
    assert(np.all(btest < bfac))
    # check interpolation in cos(theta)
    assert(np.all(pc(100,'FRONT',1) > ftest))
    assert(np.all(pc(100,'BACK',1) > btest))

def aeff_corr(pcos,acos):
    """ Need to apply a 2D correction."""
    rvals = np.ones_like(pcos)
    corrections = [
           [[0.00,0.25],[[0.35,0.7,0.99],[0.88,0.91,1.025],[0.96,0.99,0.989]]],
           [[0.25,0.50],[[0.85,0.89,1.013],[0.89,0.93,0.99],[0.99,1.00,1.03]]],
           [[0.50,0.75],[[0.35,0.57,0.975],[0.74,0.85,1.015],[0.86,0.91,0.985],[0.99,1.00,1.05]]],
           [[0.75,1.00],[[0.99,1.00,1.025]]]]
    for c in corrections:
        abounds,pcorrs = c
        amask = (acos >= abounds[0]) & (acos < abounds[1])
        idx = np.flatnonzero(amask)
        masked_pcos = pcos[amask]
        for pcorr in pcorrs:
            pmask = (masked_pcos >= pcorr[0]) & (masked_pcos < pcorr[1])
            rvals[idx[pmask]] = pcorr[2]

    # Now apply a phi correction for a range of polar angle
    mask = (pcos >= 0.6) & (pcos < 0.8) & (acos < 0.1)
    rvals[mask] *= 1.025

    return rvals

