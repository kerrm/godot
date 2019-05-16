"""
Module implements a relatively feature-complete computation of the LAT
exposure.  The purpose is primarily for fast computation of an exposure
time series for a very limited patch of sky.

This was written for Pass 7 data and has been roughly kluged to work with
Pass 8, at least commonly used event classes.

Requires: Fermi ScienceTools

author(s): Matthew Kerr
"""
import numpy as np
from astropy.io import fits
from math import sin,cos
from uw.like import pycaldb
from uw.utilities import keyword_options
from skymaps import Gti,Band,Healpix,SkyDir
import skymaps
from os.path import join
import os
from scipy.interpolate import interp1d,interp2d
from collections import deque

DEG2RAD = np.pi/180.

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
                print 'Warning, azimuth angles are wrapped to 0 to pi/2'

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

       The default implementation is fully-unbinned, i.e., when the user
       requests the livetime, the exact values for the S/C z-axis and
       zenith positions are used to calculate the incidence/zenith angles.

       This executes with comparable speed (factor of ~2 slower) to the
       Science Tools application gtltcube."""

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
        if self.verbose >= 1: print 'Processing GTI...'
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
                print 'Applied GTI mask; ontime reduced from %ds to %ds'%(
                        before,round(gti.computerOntime()))

        self.gti_starts = np.sort(gti.get_edges(True))
        self.gti_stops = np.sort(gti.get_edges(False))
        if self.verbose >= 1:
            print 'Finished computing GTI from FT1 files; total ontime = %ds'%(
                    round(gti.computeOntime()))

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
            print """"WARNING!!!
            
            GTI are present with no spacecraft pointing information!  
            This likely represents a problem with the input file. 
            This algorithm will attempt to compute the exposure correctly
            by adjusting the GTI, but you will need to apply the resulting
            mask to the events directly.
            """

            print g0[starts_in_gap]

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
        if self.verbose >= 1: print 'Loading FT2 files...'
        if not hasattr(ft2files,'__iter__'): ft2files = [ft2files]
        handles = [fits.open(ft2,memmap=False) for ft2 in ft2files]
        ft2lens = [handle['SC_DATA'].data.shape[0] for handle in handles]
        fields  = self.fields
        arrays  = [np.empty(sum(ft2lens)) for i in xrange(len(fields))]
        
        counter = 0
        for ihandle,handle in enumerate(handles):
            if self.verbose > 1:
                print '...Loading FT2 file # %d'%(ihandle)
            n = ft2lens[ihandle]
            for ifield,field in enumerate(fields):
                arrays[ifield][counter:counter+n] = handle['SC_DATA'].data.field(field)
            handle.close()
        ## TEMP? maybe.  Handle case where FT2 file is not sorted
        #starts = arrays[self.fields.index('START')]
        #a = np.argsort(starts)
        #if not (np.all(starts==starts[a])):
            #arrays = [x[a].copy() for x in arrays]
        # end TEMP
        for ifield,field in enumerate(fields):
            self.__dict__[field] = arrays[ifield]
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
        if self.verbose > 1: print 'Finished loading FT2 files!'
  
    def _process_ft2(self):
        if self.verbose >= 1: print 'Processing the FT2 file (calculating overlap with GTI)...'
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
        if self.verbose > 1: print 'Finished processing the FT2 file!'

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

    def get_cosines(self,ra,dec,theta_cut,zenith_cut,get_phi=False):
        """ Return the cosine of the arclength between the specified 
            direction and the S/C z-axis and [optionally] the azimuthal
            orientation as a cosine.

        ra -- right ascention (radians)
        dec -- declination (radians)
        theta_cut -- cosine(theta_max)
        zenith_cut -- cosine(zenith_max)

        Returns:
        mask -- True if the FT2 interval satisfies the specified cuts
        pcosines -- cosines of polar angles
        acosines -- cosines of azimuthal angles [optional]
        """    
        ra_s,ra_z = self.RA_SCZ,self.RA_ZENITH
        cdec,sdec = cos(dec),sin(dec)
        # cosine(polar angle) of source in S/C system
        pcosines  = self.COS_DEC_SCZ*cdec*np.cos(ra-self.RA_SCZ) + self.SIN_DEC_SCZ*sdec
        mask = pcosines >= theta_cut
        if zenith_cut > -1:
            zcosines = self.COS_DEC_ZENITH*cdec*np.cos(ra-self.RA_ZENITH) + self.SIN_DEC_ZENITH*sdec
            mask = mask & (zcosines>=zenith_cut)
        pcosines = pcosines[mask]
        if get_phi:
            ra_s = self.RA_SCX[mask]
            acosines = self.COS_DEC_SCX[mask]*cdec*np.cos(ra-self.RA_SCX[mask]) + self.SIN_DEC_SCX[mask]*sdec
            np.clip(np.abs(acosines/(1-pcosines**2)**0.5),0,1,out=acosines) # fold to 0-pi/2
        else: acosines = None
        return mask,pcosines,acosines

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
        print 'gti mask: %d/%d'%(mask.sum(),len(mask))
        return mask

class BinnedLivetime(Livetime):
    """See remarks for Livetime class for general information.
       
       This class provides an implementation of the livetime calculation
       in which the FT2 entries for the S/Z z-axis and zenith positions
       are binned onto a Healpix grid, allowing for a faster calculation
       with long FT2 files.
    """

    def finish(self):
        hp = Healpix(self.nside,Healpix.RING,SkyDir.EQUATORIAL)
        ras,decs = np.asarray( [hp.py_pix2ang(i) for i in xrange(12*self.nside**2)]).transpose()
        self.COS_HP_DEC = np.cos(decs)
        self.SIN_HP_DEC = np.sin(decs)
        self.HP_RA = ras
        ra_s,dec_s = self.RA_SCZ,self.DEC_SCZ
        ra_z,dec_z = self.RA_ZENITH,self.DEC_ZENITH
        self.S_PIX = np.fromiter((hp.py_ang2pix(ra_s[i],dec_s[i]) for i in xrange(len(ra_s))),dtype=int)
        self.Z_PIX = np.fromiter((hp.py_ang2pix(ra_z[i],dec_z[i]) for i in xrange(len(ra_z))),dtype=int)

    def __init__(self,nside=59,*args,**kwargs):
        self.nside = nside
        super(BinnedLivetime,self).__init__(*args,**kwargs)

    def get_cosines(self,skydir):
        ra,dec    = N.radians([skydir.ra(),skydir.dec()])

        # calculate the arclengths to the various Healpix
        cosines  = self.COS_HP_DEC*cos(dec)*np.cos(ra-self.HP_RA) + self.SIN_HP_DEC*sin(dec)
        scosines = cosines[self.S_PIX]
        if self.zenithcut > -1:
            zcosines = cosines[self.Z_PIX]
            mask = (scosines>=self.fovcut) & (zcosines>=self.zenithcut)
        else:
            mask = (scosines>=self.fovcut)
        return scosines,mask

class EfficiencyCorrection(object):
    """ Apply a trigger-rate dependent correction to the livetime."""
    
    def p(self,logE,v):
        a0,b0,a1,logEb1,a2,logEb2 = v
        b1 = (a0 - a1)*logEb1 + b0
        b2 = (a1 - a2)*logEb2 + b1
        if logE < logEb1:
            return a0*logE + b0
        if logE < logEb2:
            return a1*logE + b1
        return a2*logE + b2

    def _set_parms(self,irf_file):
        f = fits.open(irf_file)
        try:
            self.v1,self.v2,self.v3,self.v4 = f['EFFICIENCY_PARAMS'].data.field('EFFICIENCY_PARS')
        except KeyError:
            print 'Efficiency parameters not found in %s.'%irf_file
            print 'Assuming P6_v3_diff parameters.'
            self.v1 = [-1.381,  5.632, -0.830, 2.737, -0.127, 4.640]  # p0, front
            self.v2 = [ 1.268, -4.141,  0.752, 2.740,  0.124, 4.625]  # p1, front
            self.v3 = [-1.527,  6.112, -0.844, 2.877, -0.133, 4.593]  # p0, back
            self.v4 = [ 1.413, -4.628,  0.773, 2.864,  0.126, 4.592]  # p1, back

    def __init__(self,irf='P7SOURCE_V6',e=1000):
        cdbm = pycaldb.CALDBManager(irf)
        #ct0_file,ct1_file = get_irf_file(irf)
        ct0_file,ct1_file = cdbm.get_aeff()
        self._set_parms(ct0_file)
        self.set_p(e)

    def set_p(self,e):
        loge = np.log10(e)
        for key,vec in zip(['p0f','p1f','p0b','p1b'],[self.v1,self.v2,self.v3,self.v4]):
            self.__dict__[key] = self.p(loge,vec)

    def get_efficiency(self,livetime_fraction,conversion_type=0):
        p0,p1 = (self.p0f,self.p1f) if conversion_type==0 else (self.p0b,self.p1b)
        return p0*livetime_fraction + p1

class InterpTable(object):
    def __init__(self,xbins,ybins,augment=True):
        """ Interpolation bins in energy and cos(theta)."""
        self.xbins_0,self.ybins_0 = xbins,ybins
        self.augment = augment
        if augment:
            x0 = xbins[0] - (xbins[1]-xbins[0])/2
            x1 = xbins[-1] + (xbins[-1]-xbins[-2])/2
            y0 = ybins[0] - (ybins[1]-ybins[0])/2
            y1 = ybins[-1] + (ybins[-1]-ybins[-2])/2
            self.xbins = np.concatenate(([x0],xbins,[x1]))
            self.ybins = np.concatenate(([y0],ybins,[y1]))
        else:
            self.xbins = xbins; self.ybins = ybins
        self.xbins_s = (self.xbins[:-1]+self.xbins[1:])/2
        self.ybins_s = (self.ybins[:-1]+self.ybins[1:])/2

    def augment_data(self,data):
        """ Build a copy of data with outer edges replicated."""
        d = np.empty([data.shape[0]+2,data.shape[1]+2])
        d[1:-1,1:-1] = data
        d[0,1:-1] = data[0,:]
        d[1:-1,0] = data[:,0]
        d[-1,1:-1] = data[-1,:]
        d[1:-1,-1] = data[:,-1]
        d[0,0] = data[0,0]
        d[-1,-1] = data[-1,-1]
        d[0,-1] = data[0,-1]
        d[-1,0] = data[-1,0]
        return d

    def set_indices(self,x,y,bilinear=True):
        if bilinear and (not self.augment):
            print 'Not equipped for bilinear, going to nearest neighbor.'
            bilinear = False
        self.bilinear = bilinear
        if not bilinear:
            i = np.searchsorted(self.xbins,x)-1
            j = np.searchsorted(self.ybins,y)-1
        else:
            i = np.searchsorted(self.xbins_s,x)-1
            j = np.searchsorted(self.ybins_s,y)-1
        self.indices = i,j

    def value(self,x,y,data):
        i,j = self.indices
        # NB transpose here
        if not self.bilinear: return data[j,i]
        x2,x1 = self.xbins_s[i+1],self.xbins_s[i]
        y2,y1 = self.ybins_s[j+1],self.ybins_s[j]
        f00 = data[j,i]
        f11 = data[j+1,i+1]
        f01 = data[j+1,i]
        f10 = data[j,i+1]
        norm = (x2-x1)*(y2-y1)
        return ( (x2-x)*(f00*(y2-y)+f01*(y-y1)) + (x-x1)*(f10*(y2-y)+f11*(y-y1)) )/norm

    def __call__(self,x,y,data,bilinear=True,reset_indices=True):
        if reset_indices:
            self.set_indices(x,y,bilinear=bilinear)
        return self.value(x,y,data)

class EffectiveArea(object):

    defaults = (
        #('irf','P6_v3_diff','IRF to use'),
        #('irf','P7SOURCE_V6','IRF to use'),
        ('irf','P8R2_SOURCE_V6','IRF to use'),
        ('CALDB',None,'path to override environment variable'),
        ('use_phidep',False,'use azmithual dependence for effective area')
        )

    @keyword_options.decorate(defaults)
    def __init__(self,**kwargs):
        keyword_options.process(self,kwargs)
        #ct0_file,ct1_file = get_irf_file(self.irf,CALDB=self.CALDB)
        cdbm = pycaldb.CALDBManager(self.irf)
        ct0_file,ct1_file = cdbm.get_aeff()
        self._read_aeff(ct0_file,ct1_file)
        if self.use_phidep:
            self._read_phi(ct0_file,ct1_file)

    def _read_file(self,filename,tablename,columns):
        hdu = fits.open(filename); table = hdu[tablename]
        cbins = np.append(table.data.field('CTHETA_LO')[0],table.data.field('CTHETA_HI')[0][-1])
        ebins = np.append(table.data.field('ENERG_LO')[0],table.data.field('ENERG_HI')[0][-1])
        images = [np.asarray(table.data.field(c)[0],dtype=float).reshape(len(cbins)-1,len(ebins)-1) for c in columns]
        hdu.close()
        return ebins,cbins,images

    def _read_aeff(self,ct0_file,ct1_file):
        try:
            ebins,cbins,feffarea = self._read_file(ct0_file,'EFFECTIVE AREA',['EFFAREA'])
            ebins,cbins,beffarea = self._read_file(ct1_file,'EFFECTIVE AREA',['EFFAREA'])
        except KeyError:
            ebins,cbins,feffarea = self._read_file(ct0_file,'EFFECTIVE AREA_FRONT',['EFFAREA'])
            ebins,cbins,beffarea = self._read_file(ct1_file,'EFFECTIVE AREA_BACK',['EFFAREA'])
        self.ebins,self.cbins = ebins,cbins
        self.feffarea = feffarea[0]*1e4;self.beffarea = beffarea[0]*1e4
        self.aeff = InterpTable(np.log10(ebins),cbins)
        self.faeff_aug = self.aeff.augment_data(self.feffarea)
        self.baeff_aug = self.aeff.augment_data(self.beffarea)

    def _read_phi(self,ct0_file,ct1_file):
        try:
            ebins,cbins,fphis = self._read_file(ct0_file,'PHI_DEPENDENCE',['PHIDEP0','PHIDEP1'])
            ebins,cbins,bphis = self._read_file(ct1_file,'PHI_DEPENDENCE',['PHIDEP0','PHIDEP1'])
        except KeyError:
            ebins,cbins,fphis = self._read_file(ct0_file,'PHI_DEPENDENCE_FRONT',['PHIDEP0','PHIDEP1'])
            ebins,cbins,bphis = self._read_file(ct1_file,'PHI_DEPENDENCE_BACK',['PHIDEP0','PHIDEP1'])
        self.fphis = fphis; self.bphis = bphis
        self.phi = InterpTable(np.log10(ebins),cbins,augment=False)

    def _phi_mod(self,e,c,event_class,phi):
        # assume phi has already been reduced to range 0 to pi/2
        if phi is None: return 1
        tables = self.fphis if event_class==0 else self.bphis
        par0 = self.phi(e,c,tables[0],bilinear=False)
        par1 = self.phi(e,c,tables[1],bilinear=False,reset_indices=False)
        norm = 1. + par0/(1. + par1)
        phi = 2*abs((2./np.pi)*phi - 0.5)
        return (1. + par0*phi**par1)/norm

    def __call__(self,e,c,phi=None,event_class=-1,bilinear=True):
        """ Return bilinear (or nearest-neighbour) interpolation.
            
            Input:
                e -- bin energy; potentially array
                c -- bin cos(theta); potentially array

            NB -- if e and c are both arrays, they must be of the same
                  size; in other words, no outer product is taken
        """
        e = np.log10(e)
        at = self.aeff
        if event_class == -1:
            return (at(e,c,self.faeff_aug,bilinear=bilinear)*self._phi_mod(e,c,0,phi),
                    at(e,c,self.baeff_aug,bilinear=bilinear,reset_indices=False)*self._phi_mod(e,c,1,phi))
        elif event_class == 0:
            return at(e,c,self.faeff_aug)*self._phi_mod(e,c,0,phi)
        return at(e,c,self.baeff_aug)*self._phi_mod(e,c,1,phi)

    def get_file_names(self):
        return self.ct0_file,self.ct1_file

class Exposure(object):
    """ Integrate effective area and livetime over incidence angle."""

    def __init__(self,ft2file,ft1files):
        self.ft2file = ft2file
        self.ea = EffectiveArea()
        self.lt = Livetime(ft2file,ft1files)

    def __call__(self,skydir,energies,event_class=-1,theta_cut=0.4,zenith_cut=-1):
        binning = Binning(theta_bins=self.ea.cbins)
        lt = self.lt(skydir,binning,theta_cut=theta_cut,zenith_cut=zenith_cut)
        e_centers = np.asarray(energies)
        c_centers = (lt[1][1:]+lt[1][:-1])/2.

        #vals = self.ea(e_centers,(self.ea.cbins[:-1]+self.ea.cbins[1:])/2)
        vals = np.empty([2 if event_class < 0 else 1,len(e_centers),len(c_centers)])
        for i,ic in enumerate(c_centers):
            vals[...,i] = self.ea(e_centers,ic,event_class=event_class)
        vals *= lt[0]
        return vals.sum(axis=-1)

    def change_IRF(self,frontfile = None):
        self.ea = EffectiveArea(frontfile = frontfile)
        self.energies = self.event_class = None

class ExposureSeries(object):
    """ Encapsulate a time series of the exposures."""

    def __init__(self,t1,t2,exposures,normalize=True):
        """ t1 -- left edges of time series bins
            t2 -- right edges of time series bins
            exposures -- exposures for the bins; each entry may be an array
                         over energy bins
        """
        self.t1 = t1; self.t2 = t2
        self.times = (t1+t2)/2
        self.t0 = (t2[-1]-t1[0])/2+1.23456789 # ad hoc epoch
        self.exposures = exposures
        if normalize:
            for i in xrange(exposures.shape[0]):
                exposures[i] /= exposures[i].sum()
        self.sorting = np.arange(len(self.times))

    def get_mask(self,photon_times):
        idx = np.searchsorted(self.t2,photon_times)
        return photon_times >= self.t1[idx]

    def __iter__(self):
        self.counter = -1
        return self

    def next(self):
        self.counter += 1
        if self.counter == len(self.exposures):
            raise StopIteration
        return (self.exposures[self.counter])[self.sorting]

    #def get_tstarts(self): return self.t1
    #def get_tstops(self): return self.t2
    def get_times(self): return self.times
    def get_exposures(self,e): return self.exposures
    def get_t0(self): return self.t0
    def set_period(self,period,t0=None):
        """ period, t0 in seconds; alter the phases, etc. to make
            appropriate for cdf use."""
        t0 = t0 or self.get_t0()
        phi = np.mod((self.get_times()-t0)/period,1)
        self.sorting = np.argsort(phi)
        return phi[self.sorting]
        # potentially faster - requires sorted times
        #t  = self.get_times()
        #p  = (t - t0)/period
        #p += abs(int(p[0])) + 1
        #p -= p.astype(int)
        #return p

class ExposurePhase(object):
    """ Compute phase of time intervals in FT2 file.  Primarily for 
        building a properly phased orbital lightcurve.  (Accounts for 
        barycentering...)
    """

    def __init__(self,times,par,ft2='/edata/ft2.fits',output='ophase.fits',
                 clobber=True,cleanup=False):
        if clobber or (not os.path.exists(output)):
            # write a temporary file with a TIME column
            c = fits.Column(name='TIME',format='E',array=times)
            tbhdu = fits.new_table([c],header=None)
            tbhdu.writeto(output,clobber=True)
            cmd = 'tempo2 -gr fermi -ft1 %s -ft2 %s -f %s -phase -ophase'%(output,ft2,par)
            print 'Executing: \n',cmd
            os.system(cmd)
        f = fits.open(output)
        self.phase = f[1].data.field('ORBITAL_PHASE')
        f.close()
        if cleanup:
            os.remove(output)

    def __call__(self):
        return self.phase

class ExposureSeriesFactory(object):
    """A helper class to generate a simple exposure (evaluated at a
       single energy) time series for use in looking for periodic signals
       from sources with periods long enough to warrant exposure correction.
    """

    defaults = (
        ('irf','P7SOURCE_V6','IRF to use'),
        ('zenith_cut',-1,'cut on cosine(zenith angle)'),
        ('theta_cut',0.2,'cut on cosine(incidence angle)'),
        ('eff_corr',True,'if True, apply correction for ghost events'),
        ('use_phidep',True,'if True, apply azimuthal correction to effective area'),
        ('lt_kwargs',{},'additional kwargs for Livetime object'),
        ('ea_kwargs',{},'additional kwargs for EffectiveArea object'),
        ('deadtime',False,'accumulate deadtime instead of livetime')
    )

    @keyword_options.decorate(defaults)
    def __init__(self,ft1files,ft2files,**kwargs):
        keyword_options.process(self,kwargs)
        self.lt_kwargs.update({'deadtime':self.deadtime})
        self.ea_kwargs.update({'use_phidep':self.use_phidep,'irf':self.irf})
        self.lt = Livetime(ft2files,ft1files,**self.lt_kwargs)
        self.ea = EffectiveArea(**self.ea_kwargs)

    def get_livetime(self,energy,event_class=-1):
        if self.eff_corr:
            ec = EfficiencyCorrection(self.irf,energy)
            lt0 = self.lt.LIVETIME*ec.get_efficiency(self.lt.LTFRAC,0)
            lt1 = self.lt.LIVETIME*ec.get_efficiency(self.lt.LTFRAC,1)
        else:
            lt0 = lt1 = self.lt.LIVETIME
        if event_class < 0:
            return lt0,lt1
        elif event_class == 0: return lt0
        return lt1

    def get_series(self,skydir,energies):
        """ NB -- energy can be a vector or a scalar.
            NB -- returns exposures summed over event class."""
        if not hasattr(energies,'__iter__'): energies = [energies]
        ra,dec = DEG2RAD*skydir.ra(),DEG2RAD*skydir.dec()
        mask,pcosines,acosines = self.lt.get_cosines(ra,dec,self.theta_cut,self.zenith_cut,self.use_phidep)
        if acosines is not None:
            acosines = np.arccos(acosines)
        exposures = np.empty([len(energies),mask.sum()])
        for ien,en in enumerate(energies):
            lt0,lt1 = self.get_livetime(en)
            if not self.deadtime:
                aeff = self.ea(en,pcosines,phi=acosines)
                #exposures.append(aeff[0]*lt0[mask]+aeff[1]*lt1[mask])
                exposures[ien,:] = aeff[0]*lt0[mask]+aeff[1]*lt1[mask]
            else:
                #exposures.append(self.lt.LIVETIME[mask])
                exposure[ien,:] = self.lt.LIVETIME[mask]
        return ExposureSeries(self.lt.START[mask],self.lt.STOP[mask],exposures)

def image(ea,event_class=-1,logea=False,fig_base=2,ctheta=0.99,show_image=False):

    self = ea
    if event_class < 0: effarea = self.feffarea + self.beffarea
    elif event_class == 0: effarea = self.feffarea
    else: effarea = self.beffarea
    ebins,cbins = self.ebins,self.cbins

    import pylab as pl

    if show_image:
        #Generate a pseudo-color plot of the full effective area
        pl.figure(fig_base);pl.clf()
        pl.gca().set_xscale('log')
        if logea: pl.gca().set_yscale('log')
        pl.pcolor((ebins[:-1]*ebins[1:])**0.5,(cbins[:-1]+cbins[1:])/2.,effarea.reshape(len(cbins)-1,len(ebins)-1))
        pl.title('Effective Area')
        pl.xlabel('$\mathrm{Energy\ (MeV)}$')
        pl.ylabel('$\mathrm{cos( \theta)}$')
        cb = pl.colorbar()
        cb.set_label('$\mathrm{Effective\ Area\ (m^2)}$')

    #Generate a plot of the on-axis effective area with and without interpolation
    energies = np.logspace(np.log10(ebins[0]),np.log10(ebins[-1]),8*(len(ebins)-1)+1)
    f_vals,b_vals = np.array([self(e,ctheta,bilinear=True) for e in energies]).transpose()
    f_ea = skymaps.EffectiveArea('%s_front'%(ea.irf))
    b_ea = skymaps.EffectiveArea('%s_back'%(ea.irf))
    check_fvals = np.asarray([f_ea(e,ctheta) for e in energies])
    check_bvals = np.asarray([b_ea(e,ctheta) for e in energies])
    pl.figure(fig_base+2);pl.clf()
    pl.gca().set_xscale('log')
    if logea: pl.gca().set_yscale('log')
    pl.plot(energies,f_vals,label='front bilinear interp.',color='blue')
    pl.plot(energies,check_fvals,color='k',ls='--')
    pl.plot(energies,b_vals,label='back bilinear interp.',color='red')
    pl.plot(energies,check_bvals,color='k',ls='--')
    f_vals,b_vals = np.array([self(e,ctheta,bilinear=False) for e in energies]).transpose()
    pl.plot(energies,f_vals,label='front nearest-neighbour interp.',color='blue')
    pl.plot(energies,b_vals,label='back nearest-neighbour interp.',color='red')
    pl.title('On-axis Effective Area')
    pl.xlabel('$\mathrm{Energy\ (MeV)}$')
    pl.ylabel('$\mathrm{Effective\ Area\ (cm^2)}$')
    pl.legend(loc = 'lower right')
    pl.grid()

def test():
    import pylab as pl
    xb = np.linspace(0,10,11)
    yb = np.linspace(0,10,11)
    data = np.arange(0.5,100).reshape(10,10)
    #data = np.empty([10,10])
    #for i in xrange(10):
    #    data[i,:5] = i
    #    data[i,5:] = i+0.5
    ip2 = InterpTable(xb,yb)
    data_aug = ip2.augment_data(data)
    dom = np.linspace(0,10,1000)
    cod = np.empty([len(dom),len(dom)])
    for i in xrange(len(dom)):
        cod[i,:] = ip2([dom[i]]*len(dom),dom,data_aug.transpose())
    pl.figure(1);pl.clf();pl.imshow(data,interpolation='nearest')
    pl.figure(2);pl.clf();pl.imshow(data,interpolation='bilinear')
    pl.figure(3);pl.clf();pl.imshow(cod,interpolation='nearest',origin='upper')
    pl.axvline(500,color='k')
    for i in xrange(len(xb)):
        pl.axhline(i*100,color='k')
    pl.axis([0,1000,1000,0])

