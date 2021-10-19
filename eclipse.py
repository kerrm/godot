import core; reload(core)
from load_data import get_data

import numpy as np
from collections import deque


# New code for eclipse stuff

# first off the bat, let's try to simulate an eclipse with a fairly bright
# source, say J1231-1411
#data = get_data('j1231',clobber=False)

def get_simulated_timeseries(data,freq=1./(2*np.pi*3600),
        alpha=0,ph0=0,theta=0.05):
    """ Returned a CellTimeSeries object with a notch-like modulation added
        to the source photons.

        freq: "orbital" frequency for eclipse [default period 2pi hours]
        alpha: depth of eclipse. 0 = full eclipse [default], 1 = no eclipse
        ph0: left edge of eclipse [default=0]
        theta: width of eclipse [default=0.05]
    """
    cells = data.get_cells(tcell=150,time_series_only=False,
            trim_zero_exposure=True,use_barycenter=False)

    # TODO -- revise cells to know about barycentering

    # So, take superset of all cells, and the weights from them.  For each
    # cell, calculate the appropriate source and background fraction.
    # Resample the weights to determine source and background, and then 
    # redistribute them.  There will be slop within the cell itself, but
    # perhaps not a big deal.
    weights = np.concatenate([c.we for c in cells])
    src_mask = np.random.rand(len(weights)) <= weights
    nsrc = np.sum(src_mask)
    nbkg = len(src_mask)-nsrc
    tstarts = np.asarray([c.tstart for c in cells])
    tstops = np.asarray([c.tstop for c in cells])
    tmid = 0.5*(tstarts+tstops)

    # set notch frequency and functional form
    # we are adopting lambda1 = alpha, lambda2 = (1-alpha*theta)/(1-theta)
    ph = np.mod((tmid-tmid[0])*freq-ph0,1)
    src = np.where(ph < theta,alpha,(1-alpha*theta)/(1-theta))

    exp = np.asarray([c.exp for c in cells])
    src_prob = np.cumsum(exp*src)
    src_prob *= 1./src_prob[-1]
    bkg_prob = np.cumsum(exp)
    bkg_prob *= 1./bkg_prob[-1]

    src_cell_idx = np.searchsorted(src_prob,np.random.rand(nsrc))
    bkg_cell_idx = np.searchsorted(bkg_prob,np.random.rand(nbkg))

    # reassign weights to cells; super klugey atm
    src_weights = weights[src_mask]
    bkg_weights = weights[~src_mask]

    for cell in cells:
        cell.we = deque()

    for idx,w in zip(src_cell_idx,src_weights):
        cells[idx].we.append(w)

    for idx,w in zip(bkg_cell_idx,bkg_weights):
        cells[idx].we.append(w)

    for cell in cells:
        cell.we = np.asarray(cell.we)

    # OK, now we need to get this back to a timeseries object we can 
    # operate on; first, fill in any gaps
    dt = cells[0].tstop-cells[0].tstart
    newn = int((cells[-1].tstop-cells[0].tstart)/dt)
    starts = np.arange(newn)*dt+cells[0].tstart
    cell_starts = np.asarray([c.tstart for c in cells])
    cell_sexp = np.asarray([c.exp for c in cells])
    stops = starts + dt
    indices = np.searchsorted(stops,cell_starts)
    exp = np.zeros(newn)
    sexp = np.zeros(newn)
    bexp = np.zeros(newn)
    exp[indices] = cell_sexp*(data.E/data.S)
    sexp[indices] = cell_sexp
    bexp[indices] = cell_sexp*(data.B/data.S)
    cts = np.zeros(newn)
    weights = np.zeros(newn)
    weights2 = np.zeros(newn)
    for iind,ind in enumerate(indices):
        c = cells[iind]
        if len(c.we) > 0:
            cts[ind] = len(c.we)
            weights[ind] = np.sum(c.we)
            weights2[ind] = np.sum(c.we**2)

    timeseries = core.CellTimeSeries(
            starts,stops,exp,sexp,bexp,cts,weights,weights2)

    return timeseries

def get_spectrum(tc,ts,ce,se,th0,th1,nharm=40,
        DM_mem=None,MM_mem=None,tmp_mem=None,logl_mem=None):
    """ tc = cos_amps*cos_err
        ts = sin_amps*sin_err
        ce = cos_err
        se = sin_err
    """
    # analytic form for notch model Fourier coefficients; defined in such a way
    # that cos_amp = C0 + alpha*C1
    # that sin_amp = S0 + alpha*S1
    # and NB that C0 = -C1 etc.

    freqs = np.arange(1,nharm+1)*(2*np.pi)
    theta = (th1-th0)
    cos_mod = 2./freqs*(np.sin(freqs*th1)-np.sin(freqs*th0))/(1-theta)
    C1,C0 = cos_mod,-cos_mod
    sin_mod = 2./freqs*(np.cos(freqs*th0)-np.cos(freqs*th1))/(1-theta)
    S1,S0 = sin_mod,-sin_mod

    if DM_mem is None:
        DM = np.zeros_like(tc)
    else:
        DM = DM_mem
        DM[:] = 0
    if MM_mem is None:
        MM = np.zeros_like(tc)
    else:
        MM = MM_mem
        MM[:] = 0
    if tmp_mem is None:
        tmp = np.empty_like(DM)
    else:
        tmp = tmp_mem

    for i in xrange(1,nharm+1):
        # i::i works if we *leave* the 0 component
        maxn = len(tc[i::i])
        t = tmp[:maxn]
        dm = DM[1:maxn+1]
        mm = MM[1:maxn+1]

        #DM[1:maxn+1] += C1[i-1]*tc[i::i] + S1[i-1]*ts[i::i]
        dm += np.multiply(tc[i::i],C1[i-1],out=t)
        dm += np.multiply(ts[i::i],S1[i-1],out=t)

        #MM[1:maxn+1] += C1[i-1]**2*ce[i::i] + S1[i-1]**2*se[i::i]
        mm += np.multiply(ce[i::i],C1[i-1]**2,out=t)
        mm += np.multiply(se[i::i],S1[i-1]**2,out=t)

    if logl_mem is None:
        dlogl = np.empty_like(tc)
    else:
        dlogl = logl_mem
    dlogl[:] = DM
    dlogl *= dlogl
    dlogl /= MM

    return dlogl

def get_spectrum_slice(i0,i1,tc,ts,ce,se,th0,th1,nharm=40):
    """ Return spectrum in a limited frequency slice.  
    
        Arguably it would be best to do
        this for a whole raft of coefficients since we wouldn't be memory
        limited.

        Automatically fits for best alpha.

        i0 = starting index
        i1 = stopping index (inclusive)
        tc = cos_amps*cos_err
        ts = sin_amps*sin_err
        ce = cos_err
        se = sin_err
        th0 = notch starting phase
        th1 = notch ending phase
    """

    freqs = np.arange(1,nharm+1)*(2*np.pi)
    theta = (th1-th0)
    cos_mod = 2./freqs*(np.sin(freqs*th1)-np.sin(freqs*th0))/(1-theta)
    C1,C0 = cos_mod,-cos_mod
    sin_mod = 2./freqs*(np.cos(freqs*th0)-np.cos(freqs*th1))/(1-theta)
    S1,S0 = sin_mod,-sin_mod


    nfreq = i1-i0+1 # inclusive
    logl = np.empty(nfreq)
    for i in xrange(nfreq):
        idx = i + i0
        mytc = tc[idx::idx][:nharm]
        myts = ts[idx::idx][:nharm]
        myce = ce[idx::idx][:nharm]
        myse = se[idx::idx][:nharm]
        n = len(mytc)
        DM = np.sum(mytc*C1[:n]) + np.sum(myts*S1[:n])
        MM = np.sum(myce*C1[:n]**2) + np.sum(myse*S1[:n]**2)
        logl[i] = DM**2/MM

    return logl

def get_alpha(i0,i1,tc,ts,ce,se,th0,th1,nharm=40):
    """ Return best-fit value of alpha in a limited slice.  

        i0 = starting index
        i1 = stopping index (inclusive)
        tc = cos_amps*cos_err
        ts = sin_amps*sin_err
        ce = cos_err
        se = sin_err
        th0 = notch starting phase
        th1 = notch ending phase
    """

    freqs = np.arange(1,nharm+1)*(2*np.pi)
    theta = (th1-th0)
    cos_mod = 2./freqs*(np.sin(freqs*th1)-np.sin(freqs*th0))/(1-theta)
    C1,C0 = cos_mod,-cos_mod
    sin_mod = 2./freqs*(np.cos(freqs*th0)-np.cos(freqs*th1))/(1-theta)
    S1,S0 = sin_mod,-sin_mod


    nfreq = i1-i0+1 # inclusive
    logl = np.empty(nfreq)
    for i in xrange(nfreq):
        idx = i + i0
        mytc = tc[idx::idx][:nharm]
        myts = ts[idx::idx][:nharm]
        myce = ce[idx::idx][:nharm]
        myse = se[idx::idx][:nharm]
        n = len(mytc)
        DM = np.sum(mytc*C1[:n]) + np.sum(myts*S1[:n])
        MM = np.sum(myce*C1[:n]**2) + np.sum(myse*S1[:n]**2)
        #logl[i] = DM**2/MM
        logl[i] = 1+DM/MM

    return logl

def scan_grid(tc,ts,ce,se,nharm=40):
    #NB incomplete, bring from grid_slice when mature
    width_grid = np.arange(0.01,0.51,0.02)
    phase_grid = np.arange(0,1.001,0.01)
    m1,m2,m3,m4 = np.empty((4,len(tc)))
    rvals = np.empty(len(width_grid),len(phase_grid))
    for iw,w in enumerate(width_grid):
        for ip,p in enumerate(phase_grid):
            logls = get_spectrum(tc,ts,ce,se,p,p+w,nharm=nharm,
                    DM_mem=m1,MM_mem=m2,tmp_mem=m3,logl_mem=m4)
            rvals[iw,ip] = logls.max()

def scan_grid_slice(i0,i1,tc,ts,ce,se,nharm=40,dw=0.01,dphi=0.01):
    # TODO -- rewrite with "smart grid" with dphi=W/10 or something.
    width_grid = np.arange(0.01,0.51,dw)
    phase_grid = np.arange(0,1.0,dphi)
    m1,m2,m3,m4 = np.empty((4,len(tc)))
    rvals = np.empty((len(width_grid),len(phase_grid)))
    for iw,w in enumerate(width_grid):
        for ip,p in enumerate(phase_grid):
            logls = get_spectrum_slice(i0,i1,tc,ts,ce,se,p,p+w,nharm=nharm)
            rvals[iw,ip] = logls.max()
    return rvals

def scan_grid_slice_adapt(i0,i1,tc,ts,ce,se,nharm=40,fix_phase=None,
        width_grid=None):
    if width_grid is None:
        width_grid = np.logspace(-2,-0.3,51)
    #width_grid = np.arange(0.01,0.51,0.02)
    widths = deque()
    phases = deque()
    rlogls = deque()
    for iw,w in enumerate(width_grid):
        if fix_phase is not None:
            phase_grid = [fix_phase]
        else:
            phase_grid = np.arange(0,1,0.1*w)
        for ip,p in enumerate(phase_grid):
            logls = get_spectrum_slice(i0,i1,tc,ts,ce,se,p,p+w,nharm=nharm)
            rlogls.append(logls)
            widths.append(w)
            phases.append(p)
    return np.asarray(widths),np.asarray(phases),np.asarray(rlogls)
