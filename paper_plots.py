from __future__ import print_function
import core
from core import mjd2met,met2mjd
from load_data import get_data
import pylab as pl
import numpy as np
from scipy.stats import chi2,norm

def set_rcParams(ticklabelsize='medium',bigticks=False):
    import matplotlib
    try:
        pass
        #matplotlib.rcParams.pop('font.cursive')
    except KeyError:
        pass
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['DejaVu Serif'] + matplotlib.rcParams['font.serif']
    # NB -- this is a kluge; by default, mathtext.cal points to cursive,
    # so ideally we'd find the right font to put in here; but for now, do
    # this to prevent the really annoying error messages
    matplotlib.rcParams['font.cursive'] = ['DejaVu Serif'] + matplotlib.rcParams['font.cursive']
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    # FreeSerif is similar to Adobe Times
    #matplotlib.rcParams['font.serif'] = ['FreeSerif'] + matplotlib.rcParams['font.serif']
    matplotlib.rcParams['xtick.major.pad'] = 6
    matplotlib.rcParams['xtick.major.size'] = 6
    matplotlib.rcParams['xtick.minor.pad'] = 4
    matplotlib.rcParams['xtick.minor.size'] = 4
    matplotlib.rcParams['ytick.major.pad'] = 4
    matplotlib.rcParams['ytick.major.size'] = 6
    matplotlib.rcParams['ytick.minor.pad'] = 4
    matplotlib.rcParams['ytick.minor.size'] = 3
    matplotlib.rcParams['xtick.labelsize'] = ticklabelsize
    matplotlib.rcParams['ytick.labelsize'] = ticklabelsize
    matplotlib.rcParams['ps.usedistiller'] = 'xpdf'
    matplotlib.rcParams['axes.labelsize'] = 'large'
    #matplotlib.rcParams['ps.fonttype'] = 42
    #matplotlib.rcParams['pdf.fonttype'] = 42
    #matplotlib.font_manager.warnings.filterwarnings(
        #'once',message='not found')

# THIS IS FIGURE 1
# a 1-day resolution light curve for Geminga
def make_geminga_plot_first(data=None,ax=None,pulls_ax=None):
    if data is None:
        data = get_data('j0633',clobber=False)
    cells_1d =  data.get_cells(tcell=86400,use_barycenter=False,
            minimum_fractional_exposure=0.1)
    clls_1d = core.CellsLogLikelihood(cells_1d,profile_background=False)

    rvals_1d = clls_1d.get_lightcurve(tsmin=9)

    if ax is None:
        pl.figure(1); pl.clf()
        pl.subplots_adjust(hspace=0,bottom=0.12,left=0.10,right=0.98,top=0.98)
        ax = pl.subplot(1,1,1)
    core.plot_clls_lc(rvals_1d,ax,scale='linear')
    if pulls_ax is None:
        return

    # this plot isn't in the paper, but an example of how to make the
    # "pulls" for the points, i.e. the error-weighted residuals
    pulls_ax.clear()
    pl.subplots_adjust(hspace=0,bottom=0.12,left=0.10,right=0.98,top=0.98)
    y = rvals_1d[:,2]
    ye = np.where(y<1,rvals_1d[:,4],rvals_1d[:,3])
    ul = rvals_1d[:,-1] == -1
    pulls = ((y-1)/ye)[~ul]
    print(np.abs(pulls).max(),len(pulls))
    pulls_ax.hist(pulls,histtype='step',bins=np.linspace(-5,5,51),density=True,lw=2);
    dom = np.linspace(-5,5,1001)
    pulls_ax.plot(dom,norm.pdf(dom))
    pulls_ax.set_xlabel('Normalized Error')
    pulls_ax.set_yscale('log')
    pulls_ax.axis([-5,5,1e-6,1])

# THIS IS FIGURE 2
def make_new_3c279_figure(fignum=1):
    """ Make version with six panels showing with and without the bkg
        estimator.  For the revised version of the paper.
    """
    tstart = tstop = None
    profile_background = False
    fignum=2

    data = get_data('3c279',clobber=False)
    cells_1d =  data.get_cells(tcell=86400,use_barycenter=False,
            tstart=tstart,tstop=tstop,minimum_fractional_exposure=0.3)
    clls_1d = core.CellsLogLikelihood(cells_1d,
            profile_background=profile_background)

    clls_1dp = core.CellsLogLikelihood(cells_1d,
            profile_background=True)

    rvals_1d,allts = clls_1d.get_lightcurve(tsmin=9,get_ts=True)
    rvals_1dp,alltsp = clls_1dp.get_lightcurve(tsmin=9,get_ts=True)

    a = np.argmin(np.abs(np.asarray([cll.cell.get_tmid() for cll in clls_1d.clls])-mjd2met(56576.6)))
    t = clls_1d.clls[a].get_flux(profile_background=profile_background)
    print('Flux/TS of solar flare:',t[0],t[1])

    pl.close(fignum)
    pl.figure(fignum,(8,4.5)); pl.clf()
    pl.subplots_adjust(hspace=0.00,left=0.10,right=0.99,top=0.98,wspace=0.00,bottom=0.12)
    # TODO -- see if we can fix the "Warning, best guess" problems in core.
    for i in xrange(6):
        ax = pl.subplot(2,3,i+1)
        if i < 3:
            rvals = rvals_1d
        else:
            rvals = rvals_1dp
        if (i == 0) or (i == 3):
            core.plot_clls_lc(rvals,ax,scale='log')
        elif (i == 1) or (i == 4):
            core.plot_clls_lc(rvals,ax,scale='log',
                    min_mjd=54750-1,max_mjd=55450+1)
        else:
            core.plot_clls_lc(rvals,ax,scale='log',
                    min_mjd=56550-1,max_mjd=57250+1)
        # turn off extra tick labels
        if i%3 != 0:
            ax.set_ylabel('')
            ax.tick_params(axis='y',labelleft=False,right=False)
        if i < 3:
            # turn off x ticks and labels
            ax.tick_params(axis='x',top=True,bottom=False,direction='in')
        else:
            ax.tick_params(axis='x',top=False,bottom=True,direction='out')

        if (i == 1) or (i == 4):
            ax.axis([54750,55450,0.1,100])
            ax.set_xticks(np.arange(1,4)*175+54750)
        if (i==2) or (i==5):
            ax.axis([56550,57250,0.1,100])
            ax.set_xticks(np.arange(1,4)*175+56550)
            if (i==2):
                ax.set_xticklabels(['']*len(ax.get_xticks()))
            ax.plot([56576.5],[4.63],'o',fillstyle='none',markersize=10,color='C3',ls='--')
            # NB TS = 342
    return allts,alltsp

# THIS IS FIGURE 3
def make_bb_trials(ntrial=100,fignum=5):
    """ Make a plot showing the false positive rate for a variety of
    scenarios.  Warning: for a reasonable number of trials, this figure
    can take a while to generate (say half an hour).
    """

    bb_priors = range(2,11)
    
    data = get_data('3c279',clobber=False)

    rvals1 = core.bb_prior_tune(data,None,orbital=True,ntrial=ntrial,
            tstart=mjd2met(57185),tstop=mjd2met(57193),
            bb_priors=bb_priors)

    rvals2 = core.bb_prior_tune(data,86400*7,ntrial=ntrial,
            tstart=mjd2met(56000),tstop=mjd2met(56000 + 7*147),
            use_barycenter=False,bb_priors=bb_priors)

    data = get_data('j1231_topo',clobber=False)

    rvals3 = core.bb_prior_tune(data,86400*7,ntrial=ntrial,
            tstart=mjd2met(56000),tstop=mjd2met(56000 + 7*147),
            use_barycenter=False,bb_priors=bb_priors)

    x = np.asarray(bb_priors)
    y1 = (rvals1[0]-1).mean(axis=1)*(1./rvals1[1])
    y2 = (rvals2[0]-1).mean(axis=1)*(1./rvals2[1])
    y3 = (rvals3[0]-1).mean(axis=1)*(1./rvals3[1])
    p1 = np.polyfit(x[:5],np.log(y1[:5]),1)
    p2 = np.polyfit(x[:5],np.log(y2[:5]),1)
    p3 = np.polyfit(x[:5],np.log(y3[:5]),1)

    pl.close(fignum)
    pl.figure(fignum,(4,4)); pl.clf()
    pl.subplots_adjust(hspace=0,bottom=0.14,top=0.98,left=0.18,right=0.96)
    ax1 = pl.subplot(1,1,1)
    ax1.set_yscale('log')
    ax1.plot(bb_priors,y1,label='3C 279 orbital',marker='o')
    ax1.plot(bb_priors,y2,label='3C 279 weekly',marker='s')
    ax1.plot(bb_priors,y3,label='PSR J1231-1411 weekly',marker='^')
    ax1.set_xlabel('Bayesian Blocks Prior Parameter')
    ax1.set_ylabel('False Positive Fraction')

    pl.legend(loc='upper right',frameon=False)

# THIS IS FIGURE 4
def make_3c279_plot(data=None,fignum=2,clobber=False):
    """ Model likelihood with waveforms and plot."""
    if data is None:
        data = get_data('3c279',clobber=clobber)
    tstart,tstop = 57185,57193
    cells_orb = data.get_contiguous_exposure_cells(
            tstart=mjd2met(tstart),tstop=mjd2met(tstop))
    print('%d cells in the orbital time series'%(len(cells_orb)))
    clls_orb = core.CellsLogLikelihood(cells_orb,profile_background=False)
    cells_1d =  data.get_cells(tcell=86400,use_barycenter=False,
            tstart=mjd2met(tstart),tstop=mjd2met(tstop))
    clls_1d = core.CellsLogLikelihood(cells_1d,profile_background=False)

    rvals_orb,rvalsbb_orb = clls_orb.plot_cells_bb(bb_prior=8,plot_raw_cells=True)
    rvals_1d,rvalsbb_1d = clls_1d.plot_cells_bb(bb_prior=8,no_bb=True)

    pl.close(fignum)
    pl.figure(fignum,(3.5,7.5)); pl.clf()
    pl.subplots_adjust(hspace=0,top=0.98,left=0.16)
    ax1 = pl.subplot(3,1,1)


    # plot the BB orb values as blue points
    ul_mask = (rvals_orb[:,-1] == -1) & (~np.isnan(rvals_orb[:,-1]))
    t = rvals_orb[ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C0',alpha=0.3,ls=' ',ms=3)
    t = rvals_orb[~ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C0',alpha=0.3,ls=' ',ms=3)

    # plot the BB orbital values as red points
    ul_mask = (rvalsbb_orb[:,-1] == -1) & (~np.isnan(rvalsbb_orb[:,-1]))
    t = rvalsbb_orb[ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C3',alpha=0.8,ls=' ',ms=5)
    t = rvalsbb_orb[~ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C3',alpha=0.8,ls=' ',ms=3)

    ax1.axis([tstart,tstop,-2,90])
    ax1.tick_params(labelbottom=False)
    ax1.set_yticks([0,20,40,60,80])
    ax1.set_ylabel('Relative Flux')

    ax2 = pl.subplot(3,1,2)

    # plot the 1-d values as green points
    ul_mask = (rvals_1d[:,-1] == -1) & (~np.isnan(rvals_1d[:,-1]))
    t = rvals_1d[ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C0',alpha=0.3,ls=' ',ms=5)
    t = rvals_1d[~ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='s',color='C2',alpha=0.7,ls=' ',ms=5)

    # plot the BB orbital values as red points
    ul_mask = (rvalsbb_orb[:,-1] == -1) & (~np.isnan(rvalsbb_orb[:,-1]))
    t = rvalsbb_orb[ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C3',alpha=0.7,ls=' ',ms=3)
    t = rvalsbb_orb[~ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C3',alpha=0.8,ls=' ',ms=3)
    ax2.tick_params(labelbottom=False)
    ax2.set_yticks([0,20,40,60,80])
    ax2.axis([tstart,tstop,-2,90])
    ax2.set_ylabel('Relative Flux')

    ax3 = pl.subplot(3,1,3)

    # plot the BB orb values as blue points
    ul_mask = (rvals_orb[:,-1] == -1) & (~np.isnan(rvals_orb[:,-1]))
    t = rvals_orb[ul_mask].transpose()
    ax3.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C0',alpha=0.3,ls=' ',ms=3)
    t = rvals_orb[~ul_mask].transpose()
    ax3.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C0',alpha=0.3,ls=' ',ms=3)

    def waveform_model(p,times):
        """ This is an ad hoc model for a 3C 279 flare, just model it as 3
            gaussians.
        """
        pedestal,p = p[0],p[1:]
        ngauss = len(p)/3
        epochs = p[::3]
        amps = p[1::3]
        widths = p[2::3]

        model = np.ones(len(times))*pedestal
        for i in xrange(ngauss):
            model += amps[i]*np.exp( -0.5*((times-epochs[i])/widths[i])**2 )

        return model

    dom = np.linspace(tstart,tstop,1001)

    # these parameters come from fmin
    pfinal_4g = [2.76309735e+00,
                 5.71877033e+04, 3.82619900e+01, 3.88680894e-01,
                 5.71884225e+04, 3.09655050e+01, 1.02169003e-01,
                 5.71892461e+04, 5.92759520e+01, 4.03390602e-01, 
                 5.71890981e+04, 3.85870110e+01, 2.47029769e-02]
    # logl = -6940.18


    pfinal_3g = [2.81823729e+00,
                 5.71876967e+04, 3.80334078e+01, 3.82110083e-01,
                 5.71884149e+04, 2.97091591e+01, 9.71476655e-02,
                 5.71892111e+04, 6.39105951e+01, 4.01600420e-01]
    # logl = -6920.28





    ax3.plot(dom,waveform_model(pfinal_3g,dom),ls='-',color='C2',lw=2,
            alpha=0.9)
    ax3.plot(dom,waveform_model(pfinal_4g,dom),ls='-',color='C1',lw=2,
            alpha=0.9)

    ax3.set_xlabel('MJD')
    ax3.set_yticks([0,20,40,60,80])
    ax3.axis([tstart,tstop,-2,90])
    ax3.set_ylabel('Relative Flux')


# THIS IS FIGURE 5 (left)
def make_geminga_pulse_profile(fignum=3,add_inset=False):

    data = core.PhaseData(['/data/kerrm/photon_data/J0633+1746_topo.fits'],
            'PSRJ0633+1746',pulse_phase_col='PULSE_PHASE',phase_shift=0.05)

    cells_100 = data.get_cells(100)
    cells_1000 = data.get_cells(1000)

    clls_100 = core.PhaseCellsLogLikelihood(cells_100)
    clls_1000 = core.PhaseCellsLogLikelihood(cells_1000)


    rvals_100 = clls_100.get_lightcurve(tsmin=9,plot_phase=True)
    rvalsbb_1000 = clls_1000.get_bb_lightcurve(tsmin=9,plot_phase=True,
            bb_prior=10)

    pl.close(fignum)
    pl.figure(fignum,(4,4)); pl.clf()
    pl.subplots_adjust(hspace=0,bottom=0.14,top=0.98,left=0.16,right=0.96)
    ax1 = pl.subplot(1,1,1)


    # plot the bin values as blue points
    ul_mask = (rvals_100[:,-1] == -1) & (~np.isnan(rvals_100[:,-1]))
    t = rvals_100[ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C0',alpha=0.8,ls=' ',ms=3)
    t = rvals_100[~ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='s',color='C0',alpha=0.8,ls=' ',ms=3)

    # plot the BB values as red points
    ul_mask = (rvalsbb_1000[:,-1] == -1) & (~np.isnan(rvalsbb_1000[:,-1]))
    t = rvalsbb_1000[ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C3',alpha=0.8,ls=' ',ms=3)
    t = rvalsbb_1000[~ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C3',alpha=0.8,ls=' ',ms=3)

    ax1.axis([0,1,0,5.3])
    ax1.set_xticks(np.linspace(0,1,6))
    ax1.set_xlabel('Pulse Phase')
    ax1.set_ylabel('Relative Flux')

    if not add_inset:
        return

    ax2 = pl.axes([0.28,0.75,0.30,0.20])

    # plot the bin values as blue points
    ul_mask = (rvals_100[:,-1] == -1) & (~np.isnan(rvals_100[:,-1]))
    t = rvals_100[ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C0',alpha=0.8,ls=' ',ms=3)
    t = rvals_100[~ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='s',color='C0',alpha=0.8,ls=' ',ms=3)

    # plot the BB values as red points
    ul_mask = (rvalsbb_1000[:,-1] == -1) & (~np.isnan(rvalsbb_1000[:,-1]))
    t = rvalsbb_1000[ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C3',alpha=0.8,ls=' ',ms=3)
    t = rvalsbb_1000[~ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C3',alpha=0.8,ls=' ',ms=3)

    ax2.axis([0.7,1,0,0.5])
    ax2.set_xticks([0.7,0.85,1.0])

# THIS IS FIGURE 5 (right)
def make_j1231_pulse_profile(fignum=4):

    ra = 15*(12+31./60 + 11.3133718/3600)
    dec = -(14 + 11./60 + 43.63638 /3600)
    data = core.PhaseData(['data/J1231-1411_topo.fits'],'PSRJ1231-1411',
            pulse_phase_col='PULSE_PHASE',phase_shift=0.15,
            max_radius=2,ra=ra,dec=dec)

    cells_100 = data.get_cells(100)
    cells_1000 = data.get_cells(1000)

    clls_100 = core.PhaseCellsLogLikelihood(cells_100)
    clls_1000 = core.PhaseCellsLogLikelihood(cells_1000)

    rvals_100 = clls_100.get_lightcurve(tsmin=9,plot_phase=True)
    rvalsbb_1000 = clls_1000.get_bb_lightcurve(tsmin=9,plot_phase=True,
            bb_prior=10)

    pl.close(fignum)
    pl.figure(fignum,(4,4)); pl.clf()
    pl.subplots_adjust(hspace=0,bottom=0.14,top=0.98,left=0.16,right=0.96)
    ax1 = pl.subplot(1,1,1)


    # plot the bin values as blue points
    ul_mask = (rvals_100[:,-1] == -1) & (~np.isnan(rvals_100[:,-1]))
    t = rvals_100[ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C0',alpha=0.8,ls=' ',ms=3)
    t = rvals_100[~ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='s',color='C0',alpha=0.8,ls=' ',ms=3)

    # plot the BB values as red points
    ul_mask = (rvalsbb_1000[:,-1] == -1) & (~np.isnan(rvalsbb_1000[:,-1]))
    t = rvalsbb_1000[ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C3',alpha=0.8,ls=' ',ms=3)
    t = rvalsbb_1000[~ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C3',alpha=0.8,ls=' ',ms=3)

    ax1.axis([0,1,-0.2,12.5])
    ax1.set_xticks(np.linspace(0,1,6))
    ax1.set_xlabel('Pulse Phase')
    ax1.set_ylabel('Relative Flux')

    ax2 = pl.axes([0.28,0.55,0.50,0.40])

    # plot the bin values as blue points
    ul_mask = (rvals_100[:,-1] == -1) & (~np.isnan(rvals_100[:,-1]))
    t = rvals_100[ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C0',alpha=0.8,ls=' ',ms=3)
    t = rvals_100[~ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='s',color='C0',alpha=0.8,ls=' ',ms=3)

    # plot the BB values as red points
    ul_mask = (rvalsbb_1000[:,-1] == -1) & (~np.isnan(rvalsbb_1000[:,-1]))
    t = rvalsbb_1000[ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C3',alpha=0.8,ls=' ',ms=3)
    t = rvalsbb_1000[~ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C3',alpha=0.8,ls=' ',ms=3)

    ax2.axis([0,1,-0.05,0.4])
    ax2.set_xticks(np.linspace(0,1,6))

# THIS IS FIGURE 6
def make_geminga_power_spectrum(fignum=7):
    # NOTES
    # relative to an earlier data set stopping at 58183, the power in the
    # fixed-background at 1 year is a bit higher, and the power in the free-
    # background at 53 d is much higher (34 vs. 23).  I thought this was
    # due to the addition of post-SADA failure data and the much less
    # homogeneous exposure variation, but it turns out it was actually due
    # to differences in weights with pointlike sky models.  The FL8Y
    # version, however, is consistent.  TODO is a more general study of
    # the dependence of the low-frequency noise on the sky model.


    data = get_data('j0633',clobber=False)
    ts = data.get_cells(tcell=300,time_series_only=True,
            trim_zero_exposure=False,use_barycenter=True)
    f,window = core.power_spectrum_fft(ts,exp_only=True)
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)


    pl.close(fignum); pl.figure(fignum,(4,4))
    pl.subplots_adjust(hspace=0.05,left=0.18,right=0.97,top=0.97,
            wspace=0.28,bottom=0.14)
    ax1 = pl.subplot(1,1,1)

    dlogl_s = np.sort(dlogl[1:])
    dlogl_nobg_s = np.sort(dlogl_nobg[1:])
    cdf = np.arange(1,len(dlogl_s)+1).astype(float)/len(dlogl_s)
    # PSD has oversampled frequencies, use a rough estimate of sample size
    n_eff = ts.exp > 30000
    ax1.plot(dlogl_s,n_eff*(cdf-chi2.cdf(dlogl_s,2)),color='C0')
    ax1.plot(dlogl_nobg_s,n_eff*(cdf-chi2.cdf(dlogl_nobg_s,2)),color='C1')
    from scipy.stats import kstwobign
    bound = kstwobign.isf(0.10)
    ax1.axhline(bound,color='k',alpha=0.5,ls='--')
    ax1.axhline(-bound,color='k',alpha=0.5,ls='--')
    bound = kstwobign.isf(0.01)
    ax1.axhline(bound,color='k',alpha=0.5,ls='-.')
    ax1.axhline(-bound,color='k',alpha=0.5,ls='-.')
    #ax1.axis([0,40,1e-8,1])
    ax1.set_xlabel('Power')
    ax1.set_ylabel(r'$\sqrt{N}\times[EDF-\Phi(x)]$')
    ax1.axis([0,40,-2.5,2.5])

    fignum += 1
    pl.close(fignum)
    pl.figure(fignum,(8,4)); pl.clf()
    pl.subplots_adjust(hspace=0.04,left=0.07,right=0.97,top=0.97,
            wspace=0.28,bottom=0.15)

    #mask = ~mask
    ax2 = pl.subplot(1,2,1)
    ybound = 40
    mask = np.ones(len(dlogl),dtype=bool)
    #ax2.plot(f[mask]*86400,dlogl[mask],alpha=0.5,color='C0')
    #ax2.plot(f[mask]*86400,dlogl_nobg[mask],alpha=0.5,color='C1')
    ax2.plot(f[mask]*86400,dlogl[mask],alpha=0.8,color='C0')
    ax2.plot(f[mask]*86400,-dlogl_nobg[mask],alpha=0.8,color='C1')
    ax2.set_xlabel('Frequency (cycles d$^{-1}$)')
    ax2.set_ylabel('Power')
    ax2.axis([0,72,-ybound,ybound])
    f0 = 1./(95.45*60)
    scale = 35./window[np.abs(f-f0)<4e-7].max()
    ax2.plot(f*86400,scale*window,color='k',alpha=0.5)
    ax2.plot(f*86400,-scale*window,color='k',alpha=0.5)
    yticks = ax2.get_yticks()
    ax2.set_yticklabels(np.abs(ax2.get_yticks()).astype(int))

    #mask = ~mask
    ax2 = pl.subplot(1,2,2)
    mask = f < 0.1
    #ax2.plot(f[mask]*86400,dlogl[mask],alpha=0.5,color='C0')
    #ax2.plot(f[mask]*86400,dlogl_nobg[mask],alpha=0.5,color='C1')
    ax2.plot(f[mask]*86400,dlogl[mask],alpha=0.8,color='C0')
    ax2.plot(f[mask]*86400,-dlogl_nobg[mask],alpha=0.8,color='C1')
    ax2.set_xlabel('Frequency (cycles d$^{-1}$)')
    ax2.set_ylabel('Power')
    ax2.axis([0,0.05,-ybound,ybound])
    f0 = 1./(95.45*60)
    scale = 35./window[np.abs(f-f0)<4e-7].max()
    ax2.plot(f*86400,scale*window,color='k',alpha=0.5)
    ax2.plot(f*86400,-scale*window,color='k',alpha=0.5)

    f_yr = 1./365
    f_prec = 1./53.5
    width = (ax2.axis()[1]-ax2.axis()[0])*0.01
    ax2.arrow(f_yr,ybound,0,-3,width=width,head_length=2,
            fc='k',ec='k',overhang=0.3)
    ax2.arrow(f_yr,-ybound,0,3,width=width,head_length=2,
            fc='k',ec='k',overhang=0.3)
    ax2.arrow(f_prec,ybound,0,-3,width=width,head_length=2,
            fc='k',ec='k',overhang=0.3)
    ax2.arrow(f_prec,-ybound,0,3,width=width,head_length=2,
            fc='k',ec='k',overhang=0.3)
    yticks = ax2.get_yticks()
    ax2.set_yticklabels(np.abs(ax2.get_yticks()).astype(int))

    return

    ia = pl.axes([0.20,0.675,0.26,0.28])
    mask = f < 0.1
    ia.plot(f[mask]*86400,dlogl[mask],alpha=0.5,color='C0')
    ia.plot(f[mask]*86400,dlogl_nobg[mask],alpha=0.5,color='C1')
    ia.plot(f[mask]*86400,scale*window[mask],color='k',alpha=0.3)
    ia.axis([0,0.05,0,50])
    ia.axvline(1./365,ymin=0.7,color='k',ls='--')
    ia.axvline(1./53.5,ymin=0.7,color='k',ls='--')
    ia.set_yticklabels('')

# THIS IS FIGURE 7
def make_j0823_power_spectrum(fignum=8):

    data = get_data('j0823',clobber=False)
    ts = data.get_cells(tcell=300,time_series_only=True,
            trim_zero_exposure=False,use_barycenter=True)
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)
    f,window = core.power_spectrum_fft(ts,exp_only=True)

    pl.close(fignum)
    pl.figure(fignum,(4,4)); pl.clf()
    pl.subplots_adjust(hspace=0,bottom=0.15,top=0.99,left=0.14,right=0.95)

    ax1 = pl.subplot(1,1,1)
    mask = f < 1./86400
    ax1.plot(f[mask]*86400,dlogl[mask],alpha=0.8)#,ls=' ',marker='.')
    ax1.plot(f[mask]*86400,-dlogl_nobg[mask],alpha=0.8)#,ls=' ',marker='.')
    ax1.set_xlabel('Frequency (cycles d$^{-1}$)')
    ax1.set_ylabel('Power')
    ax1.axis([0,0.10,-45,45])
    f_psf = 4./365
    width = (ax1.axis()[1]-ax1.axis()[0])*0.01
    ax1.arrow(f_psf,45,0,-3,width=width,head_length=2,
            fc='k',ec='k',overhang=0.3)
    ax1.arrow(f_psf,-45,0,3,width=width,head_length=2,
            fc='k',ec='k',overhang=0.3)
    yticks = ax1.get_yticks()
    ax1.set_yticklabels(np.abs(ax1.get_yticks()).astype(int))

    """
    ia = pl.axes([0.46,0.54,0.42,0.42])
    ia.plot(f*86400,dlogl,alpha=0.5)
    ia.plot(f*86400,dlogl_nobg,alpha=0.5)
    f0 = 1./(95.45*60)
    #scale = 45./window[np.abs(f-f0)<4e-7].max()
    scale = 45./5000
    ia.plot(f,window*scale,color='k',alpha=0.3)
    ia.axis([0,72,0,40])
    ia.set_yticklabels('')
    """

    pl.sca(ax1)

# THIS IS FIGURE 8
def make_ls5039_power_spectrum(fignum=10):

    data = get_data('ls5039',clobber=False)
    if (data.max_radius is not None) and (data.max_radius < 10):
        data = get_data('ls5039',clobber=True,max_radius=10)

    ts = data.get_cells(tcell=300,time_series_only=True,
            trim_zero_exposure=False,use_barycenter=True)
    f,window = core.power_spectrum_fft(ts,exp_only=True)
    scale = 50./40000
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)
    fday = f*86400

    forb = 2.963145573933919e-06
    fprec = 2.1777777777777778e-07
    freqs = np.asarray([fprec,forb,2*forb])
    corr,pows = core.get_orbital_modulation(ts,freqs)
    f2,dlogl_nobg2,dlogl2,dlogl_null2 = core.power_spectrum_fft(ts,
            exposure_correction=corr)

    add_power = np.zeros_like(dlogl_nobg2)
    for freq,p in zip(freqs,pows):
        idx = np.argmin(np.abs(f[1:]-freq))
        add_power[idx] = p

    pl.close(fignum)
    pl.figure(fignum,(8,4)); pl.clf()
    pl.subplots_adjust(hspace=0,bottom=0.16,top=0.97,left=0.10,right=0.97)

    ax1 = pl.subplot(2,3,1)
    ax1.set_yscale('log')
    fmask = fday < 20.1
    ax1.plot(fday[fmask],dlogl_nobg[fmask],alpha=0.8,color='C0')
    #ax1.plot(fday,dlogl_nobg2+add_power,alpha=0.5,color='C1')
    ax1.plot(fday[fmask],scale*window[fmask],color='k',alpha=0.5)
    ax1.set_ylabel('Power')
    ax1.axis([-0.1,20,10,2000])
    ax1.set_xticklabels(['']*len(ax1.get_xticklabels()))

    fmask = fday < 1.01
    bb = ax1.get_position()
    ia = pl.axes([bb.x0+bb.width*0.3,bb.y0+bb.height*0.45,bb.width*0.62,bb.height*0.5])
    ia.plot(fday[fmask],dlogl_nobg[fmask],alpha=0.8,color='C0')
    #ia.plot(fday[fmask],(dlogl_nobg2+add_power)[fmask],alpha=0.5,color='C1')
    ia.plot(fday[fmask],scale*window[fmask],color='k',alpha=0.5)
    #ia.set_yscale('log')
    ia.axis([0,1,0,1500])

    ax1 = pl.subplot(2,3,4)
    fmask = fday < 20.1
    ax1.set_yscale('log')
    #ax1.plot(fday,dlogl_nobg,alpha=0.5,color='C0')
    ax1.plot(fday[fmask],(dlogl_nobg2+add_power)[fmask],alpha=0.8,color='C1')
    ax1.plot(fday[fmask],scale*window[fmask],color='k',alpha=0.5)
    ax1.set_ylabel('Power')
    ax1.axis([-0.1,20,10,2000])
    ax1.set_xlabel('Frequency (cycles d$^{-1}$)')

    fmask = fday < 1.01
    bb = ax1.get_position()
    ia = pl.axes([bb.x0+bb.width*0.3,bb.y0+bb.height*0.45,bb.width*0.625,bb.height*0.5])
    ##ia.plot(fday[fmask],dlogl_nobg[fmask],alpha=0.8,color='C0')
    ia.plot(fday[fmask],(dlogl_nobg2+add_power)[fmask],alpha=0.8,color='C1')
    #ia.plot(fday[fmask],scale*window[fmask],color='k',alpha=0.5)
    ia.axis([0,1,0,1500])


    ax2 = pl.subplot(2,3,2)
    #ia = pl.axes([0.30,0.30,0.60,0.60])
    #mask = (f > 14.2/86400) & (f < 16.2/86400)
    fmask = fday < 1.01
    ybound = 50
    ax2.plot(fday[fmask],dlogl_nobg[fmask],alpha=0.8,color='C0')
    ax2.plot(fday[fmask],(scale*window)[fmask],color='k',alpha=0.5)
    ax2.axis([0,1,0,ybound])
    ax2.set_xticklabels(['']*len(ax2.get_xticklabels()))

    width = (ax2.axis()[1]-ax2.axis()[0])*0.01
    ax2.arrow(forb*86400,ybound,0,-3,width=width,head_length=2,
            fc='k',ec='k',overhang=0.3)
    ax2.arrow(2*forb*86400,ybound,0,-3,width=width,head_length=2,
            fc='k',ec='k',overhang=0.3)
    ax2.arrow(3*forb*86400,ybound,0,-3,width=width,head_length=2,
            fc='k',ec='k',overhang=0.3)

    ax2 = pl.subplot(2,3,5)

    ax2.plot(fday[fmask],(dlogl_nobg2+add_power)[fmask],alpha=0.8,color='C1')
    ax2.plot(fday[fmask],(scale*window)[fmask],color='k',alpha=0.5)
    width = (ax2.axis()[1]-ax2.axis()[0])*0.01
    ax2.arrow(forb*86400,ybound,0,-3,width=width,head_length=2,
            fc='k',ec='k',overhang=0.3)
    ax2.arrow(2*forb*86400,ybound,0,-3,width=width,head_length=2,
            fc='k',ec='k',overhang=0.3)
    ax2.arrow(3*forb*86400,ybound,0,-3,width=width,head_length=2,
            fc='k',ec='k',overhang=0.3)
    ax2.axis([0,1,ybound,0])
    ax2.set_xlabel('Frequency (cycles d$^{-1}$)')

    ax3 = pl.subplot(2,3,3)
    fmask = (fday > 14.3) & (fday < 15.9)
    ybound = 70
    scale *= 0.6
    ax3.plot(fday[fmask],dlogl_nobg[fmask],alpha=0.8,color='C0')
    ax3.plot(fday[fmask],scale*window[fmask],color='k',alpha=0.5)
    ax3.plot((f-freqs[1])[fmask]*86400,scale*window[fmask]*0.5,color='k',alpha=0.5)
    ax3.plot((f+freqs[1])[fmask]*86400,scale*window[fmask]*0.5,color='k',alpha=0.5)
    ax3.plot((f-freqs[2])[fmask]*86400,scale*window[fmask]*0.25,color='k',alpha=0.5)
    ax3.plot((f+freqs[2])[fmask]*86400,scale*window[fmask]*0.25,color='k',alpha=0.5)
    ax3.axis([14.4,15.8,0,ybound])
    ax3.set_xticklabels(['']*len(ax3.get_xticklabels()))

    ax3 = pl.subplot(2,3,6)

    ax3.plot(fday[fmask],(dlogl_nobg2+add_power)[fmask],alpha=0.8,color='C1')
    ax3.plot(fday[fmask],scale*window[fmask],color='k',alpha=0.5)
    ax3.plot((f-freqs[1])[fmask]*86400,scale*window[fmask]*0.5,color='k',alpha=0.5)
    ax3.plot((f+freqs[1])[fmask]*86400,scale*window[fmask]*0.5,color='k',alpha=0.5)
    ax3.plot((f-freqs[2])[fmask]*86400,scale*window[fmask]*0.25,color='k',alpha=0.5)
    ax3.plot((f+freqs[2])[fmask]*86400,scale*window[fmask]*0.25,color='k',alpha=0.5)
    ax3.axis([14.4,15.8,ybound,0])
    ax3.set_xlabel('Frequency (cycles d$^{-1}$)')
    #ia.set_yticklabels('')

def make_ls5039_power_comparison(fignum=11):

    data = get_data('ls5039',clobber=False)
    if (data.max_radius is not None) and (data.max_radius < 10):
        data = get_data('ls5039',clobber=True,max_radius=10)

    ts = data.get_cells(tcell=300,time_series_only=True,
            trim_zero_exposure=False,use_barycenter=True)
    f,window = core.power_spectrum_fft(ts,exp_only=True)
    scale = 50./40000
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)

    pl.close(fignum)
    pl.figure(fignum,(8,4)); pl.clf()
    pl.subplots_adjust(hspace=0,bottom=0.16,top=0.97,left=0.10,right=0.97)

    ax1 = pl.subplot(1,3,1)
    ax1.plot(f*86400,dlogl_nobg,alpha=0.5,color='C0')
    ax1.plot(f*86400,dlogl,alpha=0.5,color='C1')
    ax1.set_ylabel('Power')
    ax1.axis([0,1,0,1500])

    data = get_data('ls5039',clobber=True,do_pickle=False,max_radius=5)
    ts = data.get_cells(tcell=300,time_series_only=True,
            trim_zero_exposure=False,use_barycenter=True)
    f,window = core.power_spectrum_fft(ts,exp_only=True)
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)

    ax2 = pl.subplot(1,3,2)
    ax2.plot(f*86400,dlogl_nobg,alpha=0.5,color='C0')
    ax2.plot(f*86400,dlogl,alpha=0.5,color='C1')
    ax2.axis([0,1,0,1500])
    ax2.set_xlabel('Frequency (cycles d$^{-1}$)')

    data = get_data('ls5039',clobber=True,do_pickle=False,max_radius=2)
    ts = data.get_cells(tcell=300,time_series_only=True,
            trim_zero_exposure=False,use_barycenter=True)
    f,window = core.power_spectrum_fft(ts,exp_only=True)
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)

    ax3 = pl.subplot(1,3,3)
    ax3.plot(f*86400,dlogl_nobg,alpha=0.5,color='C0')
    ax3.plot(f*86400,dlogl,alpha=0.5,color='C1')
    ax3.axis([0,1,0,1500])

# THIS IS FIGURE 9
# NB this relies on numbers that were stored by using different versions
# of the LS 5039 data set and make_ls5039_power_comparison
def plot_ls5039_aperture_dependence(fignum=12):

    pl.close(fignum)
    pl.figure(fignum,(4,4)); pl.clf()
    pl.subplots_adjust(hspace=0,bottom=0.16,top=0.98,left=0.19,right=0.97)

    x = np.asarray([2,5,10])
    harm1_bg0 = np.asarray([1257,1479,1444])
    harm2_bg0 = np.asarray([256,286,267])
    harm1_bg1 = np.asarray([619,1072,1235])
    harm2_bg1 = np.asarray([152,212,239])

    ax1 = pl.subplot(1,1,1)
    ax1.plot(x,harm1_bg0,color='C0',ls='-',marker='o',label='Orbital Freq.')
    ax1.plot(x,harm1_bg1,color='C0',ls='--',marker='o')
    ax1.plot(x,harm2_bg0,color='C1',ls='-',marker='o',label='2x Orbital Freq.')
    ax1.plot(x,harm2_bg1,color='C1',ls='--',marker='o')
    ax1.set_xticks([2,5,10])
    ax1.set_xlabel('Aperture Radius (deg.)')
    ax1.set_ylabel('Power')

    pl.legend(loc='center right',frameon=False)

# THIS IS FIGURE 10
def make_cygx3_plot(fignum=13):
    p0 = 0.199693736062
    forb = 1./p0
    data = get_data('cygx3',clobber=False)
    cells = data.get_cells(tcell=86400*14,use_barycenter=False)
    clls = core.CellsLogLikelihood(cells,profile_background=True)

    pl.close(fignum)
    pl.figure(fignum,(8,4)); pl.clf()
    pl.subplots_adjust(hspace=0.0,wspace=0.22,bottom=0.16,top=0.97,left=0.09,right=0.98)

    ax1 = pl.subplot(1,3,1)

    # disable upper limits -- want best estimates of flux density
    r1,r2 = clls.plot_cells_bb(bb_prior=8,tsmin=-1,ax=ax1)
    ax1.axis([54450,58750,-1,30])
    ax1.set_xticks([55500,56600,57700])
    ax1.set_ylabel("Relative Flux / Power")

    left_edges = r2[:,0]-r2[:,1]
    right_edges = r2[:,0]+r2[:,1]
    edges = mjd2met(np.append(left_edges,right_edges[-1]))
    scales = r2[:,2]
    ts = data.get_cells(tcell=600,time_series_only=True,trim_zero_exposure=False,scale_series=[edges,scales])

    # power spectrum with re-scaled data
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)
    fcygx3 = 1./(0.19968476+5.42e-10*(56561-40000))
    freqs = [fcygx3/86400]
    corr,pows = core.get_orbital_modulation(ts,freqs)

    # power spectrum with rescaled data and spectral leakage reduction 
    f2,dlogl_nobg2,dlogl2,dlogl_null2 = core.power_spectrum_fft(ts,exposure_correction=corr)
    ts = data.get_cells(tcell=600,time_series_only=True,trim_zero_exposure=False)
    # power spectrum *without* re-scaling
    f3,dlogl_nobg3,dlogl3,dlogl_null3 = core.power_spectrum_fft(ts)

    add_power = np.zeros_like(dlogl_nobg2)
    for freq,p in zip(freqs,pows):
        idx = np.argmin(np.abs(f[1:]-freq))
        add_power[idx] = p

    fday = f*86400
    ax2 = pl.subplot(1,3,2)
    ax2.clear()
    ax2.plot(fday,dlogl_nobg3,alpha=0.8,color='C0') # no re-scaling
    #ax2.plot(f*86400,dlogl_nobg2+add_power,alpha=0.5,color='C1')
    #ax2.plot(f*86400,scale*window,color='k',alpha=0.3)
    ax2.plot(fday,-(dlogl_nobg2+add_power),alpha=0.8,color='C1') # rescaling with correction
    ax2.set_xlabel('Frequency (cycles d$^{-1}$)')
    #ax2.set_ylabel('Power')
    ybound = 240
    ax2.axis([-1,36,-ybound,ybound])
    width = (ax2.axis()[1]-ax2.axis()[0])*0.01
    ax2.arrow(forb,ybound,0,-21,width=width,head_length=14,
            fc='k',ec='k',overhang=0.3)
    ax2.arrow(forb,-ybound,0,21,width=width,head_length=14,
            fc='k',ec='k',overhang=0.3)


    f,window = core.power_spectrum_fft(ts,exp_only=True)
    fmask = (fday > (forb-0.06)) & (fday < (forb+0.06))
    ax3 = pl.subplot(1,3,3)
    ax3.clear()
    ax3.plot(fday[fmask],dlogl_nobg[fmask],alpha=0.8,color='C2') # rescaling without correction
    ax3.plot(fday[fmask],-(dlogl_nobg2+add_power)[fmask],alpha=0.8,color='C1')
    fmask = (fday/3 > (forb-0.06)) & (fday/3 < (forb+0.06))
    ax3.plot((fday/3)[fmask],window[fmask]/14570*50*3,color='k',alpha=0.5)
    ax3.plot((fday/3)[fmask],-window[fmask]/14570*50*3,color='k',alpha=0.5)

    ax3.set_xlabel('Frequency (cycles d$^{-1}$)')
    ax3.axis([forb-0.05,forb+0.05,-ybound,ybound])
    width = (ax3.axis()[1]-ax3.axis()[0])*0.01
    ax3.arrow(forb,ybound,0,-21,width=width,head_length=14,
            fc='k',ec='k',overhang=0.3)
    ax3.arrow(forb,-ybound,0,21,width=width,head_length=14,
            fc='k',ec='k',overhang=0.3)

    #ia = pl.axis([0.78,0.48,0.18,0.45])

set_rcParams()

#####################################################################
# some old examples/old code that aren't in the paper
# caveat emptor!
#####################################################################

def make_crab_pulse_plot(fignum=4):

    data = core.PhaseData(['data/J0534+2200_topo.fits'],'PSRJ0534+2200')

    cells_100 = data.get_cells(100)
    cells_1000 = data.get_cells(1000)

    clls_100 = core.PhaseCellsLogLikelihood(cells_100)
    clls_1000 = core.PhaseCellsLogLikelihood(cells_1000)

    rvals_100 = clls_100.get_lightcurve(tsmin=9,plot_phase=True)
    rvalsbb_1000 = clls_1000.get_bb_lightcurve(tsmin=9,plot_phase=True,
            bb_prior=10)

    pl.close(fignum)
    pl.figure(fignum,(4,4)); pl.clf()
    pl.subplots_adjust(hspace=0,bottom=0.14,top=0.98,left=0.16,right=0.96)
    ax1 = pl.subplot(1,1,1)


    # plot the bin values as blue points
    ul_mask = (rvals_100[:,-1] == -1) & (~np.isnan(rvals_100[:,-1]))
    t = rvals_100[ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C0',alpha=0.8,ls=' ',ms=3)
    t = rvals_100[~ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='s',color='C0',alpha=0.8,ls=' ',ms=3)

    # plot the BB values as red points
    ul_mask = (rvalsbb_1000[:,-1] == -1) & (~np.isnan(rvalsbb_1000[:,-1]))
    t = rvalsbb_1000[ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C3',alpha=0.8,ls=' ',ms=3)
    t = rvalsbb_1000[~ul_mask].transpose()
    ax1.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C3',alpha=0.8,ls=' ',ms=3)

    ax1.axis([0,1,-0.2,12.5])
    ax1.set_xticks(np.linspace(0,1,6))
    ax1.set_xlabel('Pulse Phase')
    ax1.set_ylabel('Relative Flux')

    ax2 = pl.axes([0.28,0.55,0.50,0.40])

    # plot the bin values as blue points
    ul_mask = (rvals_100[:,-1] == -1) & (~np.isnan(rvals_100[:,-1]))
    t = rvals_100[ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C0',alpha=0.8,ls=' ',ms=3)
    t = rvals_100[~ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='s',color='C0',alpha=0.8,ls=' ',ms=3)

    # plot the BB values as red points
    ul_mask = (rvalsbb_1000[:,-1] == -1) & (~np.isnan(rvalsbb_1000[:,-1]))
    t = rvalsbb_1000[ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C3',alpha=0.8,ls=' ',ms=3)
    t = rvalsbb_1000[~ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C3',alpha=0.8,ls=' ',ms=3)

    ax2.axis([0,1,-0.05,0.4])
    ax2.set_xticks(np.linspace(0,1,6))

def make_3c279_plot_first(ax=None,profile_background=False):

    tstart = tstop = None
    #if profile_background:
    #    tstart = core.mjd2met(56400)
    #    tstop = core.mjd2met(56900)

    data = get_data('3c279')
    cells_1d =  data.get_cells(tcell=86400,use_barycenter=False,
            tstart=tstart,tstop=tstop)
    clls_1d = core.CellsLogLikelihood(cells_1d,
            profile_background=profile_background)

    rvals_1d = clls_1d.get_lightcurve(tsmin=9)

    a = np.argmin(np.abs(np.asarray([cll.cell.get_tmid() for cll in clls_1d.clls])-mjd2met(56576.6)))
    t = clls_1d.clls[a].get_flux(profile_background=profile_background)
    print('Flux/TS of solar flare:',t[0],t[1])

    if ax is None:
        pl.figure(1); pl.clf()
        ax = pl.gca()
    core.plot_clls_lc(rvals_1d,ax,scale='log')


def make_geminga_plot_second(data=None,ax=None):
    if data is None:
        data = get_data('j0633')
    cells_1d =  data.get_cells(tcell=86400,use_barycenter=False,
            tstart=mjd2met(56000),tstop=mjd2met(56500))
    clls_1d = core.CellsLogLikelihood(cells_1d,profile_background=False)

    rvals_1d = clls_1d.get_lightcurve(tsmin=9)

    if ax is None:
        pl.figure(1); pl.clf()
        ax = pl.gca()
    #ax.set_yscale('log')
    core.plot_clls_lc(rvals_1d,ax)

def make_j2021_plot(data=None,profile_background=False):
    """ NB appears to have some flares -- associated with background?"""
    if data is None:
        data = get_data('j2021',clobber=False)
    #tstart,tstop = 56620,56670
    #tstart,tstop = 57180,57230
    #tstart,tstop = 57180,57200
    cells_1d =  data.get_cells(tcell=86400,use_barycenter=False)
    clls_1d = core.CellsLogLikelihood(cells_1d,profile_background=
        profile_background)
    cells_1m =  data.get_cells(tcell=86400*28,use_barycenter=False)
    clls_1m = core.CellsLogLikelihood(cells_1m,profile_background=
        profile_background)

    rvals_1d,rvalsbb_1d = clls_1d.plot_cells_bb(bb_prior=10,
            plot_raw_cells=True,no_bb=False)
    rvals_1m,rvalsbb_1m = clls_1m.plot_cells_bb(bb_prior=10,
            plot_raw_cells=True,no_bb=True)

    pl.figure(1); pl.clf()

    # plot the BB orb values as blue points
    ul_mask = (rvals_1d[:,-1] == -1) & (~np.isnan(rvals_1d[:,-1]))
    t = rvals_1d[ul_mask].transpose()
    pl.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C0',alpha=0.3,ls=' ',ms=3)
    t = rvals_1d[~ul_mask].transpose()
    pl.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C0',alpha=0.2,ls=' ',ms=3)

    # plot the BB orbital values as red points
    ul_mask = (rvalsbb_1d[:,-1] == -1) & (~np.isnan(rvalsbb_1d[:,-1]))
    t = rvalsbb_1d[ul_mask].transpose()
    pl.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C3',alpha=0.8,ls=' ',ms=5)
    t = rvalsbb_1d[~ul_mask].transpose()
    pl.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C3',alpha=0.8,ls=' ',ms=3)

    pl.figure(2); pl.clf()

    # plot the 1-d values as blue points
    ul_mask = (rvals_1m[:,-1] == -1) & (~np.isnan(rvals_1m[:,-1]))
    t = rvals_1m[ul_mask].transpose()
    pl.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C0',alpha=0.3,ls=' ',ms=5)
    t = rvals_1m[~ul_mask].transpose()
    pl.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='s',color='C0',alpha=0.8,ls=' ',ms=5)

    # plot the BB orbital values as red points
    ul_mask = (rvalsbb_1d[:,-1] == -1) & (~np.isnan(rvalsbb_1d[:,-1]))
    t = rvalsbb_1d[ul_mask].transpose()
    pl.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C3',alpha=0.8,ls=' ',ms=3)
    t = rvalsbb_1d[~ul_mask].transpose()
    pl.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C3',alpha=0.8,ls=' ',ms=3)

    #pl.axis([tstart,tstop,-1,85])

def make_figure_1(fignum=1):

    pl.close(fignum)
    pl.figure(fignum,(8,3)); pl.clf()
    pl.subplots_adjust(hspace=0.05,left=0.10,right=0.97,top=0.98,wspace=0.02,
            bottom=0.17)
    ax1 = pl.subplot(1,3,1)
    make_geminga_plot_first(ax=ax1)
    ax1.axis([54600,58300,5e-2,80])
    ax1.set_ylabel('Relative Flux')

    x0 = ax1.get_position().bounds[0]
    ax1_inset = pl.axes([x0 + 0.07,0.66,0.20,0.30])
    make_geminga_plot_second(ax=ax1_inset)
    ax1_inset.axis([56000,56500,0.6,1.4])
    ax1_inset.tick_params(length=3)

    ax2 = pl.subplot(1,3,2)
    make_3c279_plot_first(ax=ax2,profile_background=False)
    ax2.plot([56576.5],[22.86],'o',fillstyle='none',markersize=10,color='C3',ls='--')
    ax2.tick_params(labelleft=False,which='both',left=False,right=False)
    ax2.axis([54600,58300,5e-2,80])
    ax2.set_xlabel('MJD')

    ax3 = pl.subplot(1,3,3)
    make_3c279_plot_first(ax=ax3,profile_background=True)
    ax3.plot([56576.5],[22.86],'o',fillstyle='none',markersize=10,color='C3',ls='--')
    ax3.set_xticks([56500,56600,56700,56800])
    ax3.tick_params(labelleft=False,labelright=False,which='both',
            left=False,right=True)
    #ax3.axis([54600,58300,5e-2,80])
    ax3.axis([56400,56900,5e-2,80])

def make_j1018_plot(fignum=6):
    data = get_data('j1018',clobber=False)
    ts = data.get_cells(tcell=300,time_series_only=True,
            trim_zero_exposure=False,use_barycenter=True)
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)

    pl.figure(fignum); pl.clf()
    ax = pl.gca()
    mask = f > 1./86400
    ax.hist(dlogl_nobg[mask],histtype='step',bins=np.linspace(0,50,101),
            normed=True)
    dom = np.linspace(0,50,1001)
    ax.plot(dom,chi2.pdf(dom,2))
    ax.axis([0,50,1e-10,1])
    ax.set_xlabel('Power')

    pl.figure(fignum+1); pl.clf()
    mask = ~mask
    ax = pl.gca()
    ax.plot(f[mask]*86400,dlogl[mask],alpha=0.5)
    ax.plot(f[mask]*86400,dlogl_nobg[mask],alpha=0.5)
    ax.set_xlabel('Frequency (per day)')
    ax.set_ylabel('Power')
    ax.axvline(1./53.4,ymin=0.9,color='C3',alpha=0.3)
    for i in xrange(18):
        f0 = 1./(96*60)*86400
        ax.axvline(f0/(i+1),ymin=0.9,color='C3',alpha=0.3,ls='--')
    #ax1.axvline(1./53,ymin=0.9,color='C3')

def make_j1018_power_spectrum(fignum=9):

    data = get_data('j1018',clobber=False)
    ts = data.get_cells(tcell=300,time_series_only=True,
            trim_zero_exposure=False,use_barycenter=True)
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)

    pl.close(fignum)
    pl.figure(fignum,(4,4)); pl.clf()
    pl.subplots_adjust(hspace=0,bottom=0.16,top=0.98,left=0.18,right=0.97)

    ax1 = pl.subplot(1,1,1)
    ax1.plot(f*86400,dlogl,alpha=0.5)
    ax1.plot(f*86400,dlogl_nobg,alpha=0.5)
    ax1.set_xlabel('Frequency (cycles d$^{-1}$)')
    ax1.set_ylabel('Power')
    ax1.axis([0,72,0,300])

    ia = pl.axes([0.28,0.34,0.60,0.60])
    mask = f < 1./86400
    ia.plot(f[mask]*86400,dlogl[mask],alpha=0.5)
    ia.plot(f[mask]*86400,dlogl_nobg[mask],alpha=0.5)
    ia.axis([0,0.5,0,300])
    ia.set_yticklabels('')

def make_3c279_power_spectrum(fignum=7):

    data = get_data('3c279',clobber=False)
    ts = data.get_cells(tcell=300,time_series_only=True,
            trim_zero_exposure=False,use_barycenter=True)
    f,window = core.power_spectrum_fft(ts,exp_only=True)
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)

    #pl.close(fignum)
    #pl.figure(fignum,(8,3)); pl.clf()
    #pl.subplots_adjust(hspace=0.05,left=0.09,right=0.97,top=0.97,wspace=0.28,bottom=0.18)
#

    pl.close(fignum)
    pl.figure(fignum,(4,4)); pl.clf()
    pl.subplots_adjust(hspace=0,bottom=0.16,top=0.98,left=0.18,right=0.97)

    ax1 = pl.subplot(1,1,1)
    ax1.plot(f*86400,dlogl,alpha=0.5)
    ax1.plot(f*86400,dlogl_nobg,alpha=0.5)
    ax1.set_xlabel('Frequency (cycles d$^{-1}$)')
    ax1.set_ylabel('Power')
    ax1.axis([0,72,0,500])

    f0 = 1./(95.45*60)
    scale = 250./window[np.abs(f-f0)<1e-7].max()
    ax1.plot(f*86400,scale*window,color='k',alpha=0.3)

    ia = pl.axes([0.28,0.34,0.60,0.60])
    mask = f < 1./86400
    ia.plot(f[mask]*86400,dlogl[mask],alpha=0.5)
    ia.plot(f[mask]*86400,dlogl_nobg[mask],alpha=0.5)
    ia.axis([0,0.5,0,300])
    ia.set_yticklabels('')

