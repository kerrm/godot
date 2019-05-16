#TODO: revise this so "make figure" plots actually follow paper notation.
import core; reload(core)
from core import mjd2met,met2mjd
#from examples import get_data
from load_data import get_data
import pylab as pl
import numpy as np
from scipy.stats import chi2

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

# compare 2dg vs 5 dg
# don't see much of a difference for 1231; not clear profile helps, either!
# (put another way, variability could be "real")
# pulls distribution to show stability of Geminga (others?)

# 3C 279 detail plot showing various methods (Fig N)
def make_3c279_plot(data=None,fignum=2,clobber=False):
    if data is None:
        data = get_data('3c279',clobber=clobber)
    #tstart,tstop = 56620,56670
    #tstart,tstop = 57180,57230
    tstart,tstop = 57185,57193
    cells_orb = data.get_contiguous_exposure_cells(
            tstart=mjd2met(tstart),tstop=mjd2met(tstop))
    print '%d cells in the orbital time series'%(len(cells_orb))
    clls_orb = core.CellsLogLikelihood(cells_orb,profile_background=False)
    cells_1d =  data.get_cells(tcell=86400,use_barycenter=False,
            tstart=mjd2met(tstart),tstop=mjd2met(tstop))
    clls_1d = core.CellsLogLikelihood(cells_1d,profile_background=False)

    rvals_orb,rvalsbb_orb = clls_orb.plot_cells_bb(bb_prior=8,plot_raw_cells=True)
    rvals_1d,rvalsbb_1d = clls_1d.plot_cells_bb(bb_prior=8,no_bb=True)

    pl.close(fignum)
    #pl.figure(fignum,(3.5,5)); pl.clf()
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
    #pl.xticks(visible=False)
    ax1.tick_params(labelbottom=False)
    ax1.set_yticks([0,20,40,60,80])
    ax1.set_ylabel('Relative Flux')

    ax2 = pl.subplot(3,1,2)

    # plot the 1-d values as blue points
    ul_mask = (rvals_1d[:,-1] == -1) & (~np.isnan(rvals_1d[:,-1]))
    t = rvals_1d[ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C0',alpha=0.3,ls=' ',ms=5)
    t = rvals_1d[~ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='s',color='C0',alpha=0.8,ls=' ',ms=5)

    # plot the BB orbital values as red points
    ul_mask = (rvalsbb_orb[:,-1] == -1) & (~np.isnan(rvalsbb_orb[:,-1]))
    t = rvalsbb_orb[ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C3',alpha=0.8,ls=' ',ms=3)
    t = rvalsbb_orb[~ul_mask].transpose()
    ax2.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C3',alpha=0.8,ls=' ',ms=3)
    #ax2.set_xlabel('MJD')
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

    # these parameters come from fmin (see examples.py)
    pfinal = [2.05182844e+00,   5.71877020e+04,   3.00763371e+01,
              3.87152169e-01,   5.71884231e+04,   2.47303709e+01,
              1.04508244e-01,   5.71892509e+04,   4.65444009e+01,
              4.01144696e-01,   5.71890940e+04,   3.09453926e+01,
              2.99894899e-02]

    ax3.plot(dom,waveform_model(pfinal,dom),color='C2')

    ax3.set_xlabel('MJD')
    ax3.set_yticks([0,20,40,60,80])
    ax3.axis([tstart,tstop,-2,90])
    ax3.set_ylabel('Relative Flux')


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
    print 'Flux/TS of solar flare:',t[0],t[1]

    if ax is None:
        pl.figure(1); pl.clf()
        ax = pl.gca()
    core.plot_clls_lc(rvals_1d,ax,scale='log')

def make_geminga_plot_first(data=None,ax=None):
    if data is None:
        data = get_data('j0633',clobber=False)
    cells_1d =  data.get_cells(tcell=86400,use_barycenter=False)
    clls_1d = core.CellsLogLikelihood(cells_1d,profile_background=False)

    rvals_1d = clls_1d.get_lightcurve(tsmin=9)

    if ax is None:
        pl.figure(1); pl.clf()
        ax = pl.gca()
    #ax.set_yscale('log')
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

def make_figure_2(fignum=2):
    make_3c279_plot()
    pl.savefig('fig2.pdf')


def make_figure_3(fignum=3):

    data = core.PhaseData(['data/J0633+1746_topo.fits'],'PSRJ0633+1746',
            pulse_phase_col='PULSE_PHASE',phase_shift=0.05)

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

def make_figure_4(fignum=4):

    #data = core.PhaseData(['data/J1231-1411_topo_2dg.fits'],'PSRJ1231-1411',
    #        pulse_phase_col='PULSE_PHASE',phase_shift=0.25)
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

# NB -- new J1231 file has updated phase, incorporate later

def make_figure_5(ntrial=100,fignum=5):
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

def make_geminga_power_spectrum(fignum=7):

    data = get_data('j0633',clobber=False)
    ts = data.get_cells(tcell=300,time_series_only=True,
            trim_zero_exposure=False,use_barycenter=True)
    f,window = core.power_spectrum_fft(ts,exp_only=True)
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)

    pl.close(fignum)
    pl.figure(fignum,(8,4)); pl.clf()
    pl.subplots_adjust(hspace=0.05,left=0.09,right=0.97,top=0.97,wspace=0.28,bottom=0.18)

    ax1 = pl.subplot(1,2,2)
    ax1.set_yscale('log')

    #mask = f > 1./86400
    mask = np.ones(len(f),dtype=bool)
    ax1.hist(dlogl_nobg[mask],histtype='step',bins=np.linspace(0,50,101),
            normed=True,color='C1')
    ax1.hist(dlogl[mask],histtype='step',bins=np.linspace(0,50,101),
            normed=True,color='C0')
    dom = np.linspace(0,50,1001)
    ax1.plot(dom,chi2.pdf(dom,2),color='k')
    ax1.axis([0,40,1e-8,1])
    ax1.set_xlabel('Power')
    ax1.set_ylabel('Probability Density')
    

    #mask = ~mask
    ax2 = pl.subplot(1,2,1)
    ax2.plot(f[mask]*86400,dlogl[mask],alpha=0.5,color='C0')
    ax2.plot(f[mask]*86400,dlogl_nobg[mask],alpha=0.5,color='C1')
    ax2.set_xlabel('Frequency (cycles/day)')
    ax2.set_ylabel('Power')
    ax2.axis([0,72,0,50])
    f0 = 1./(95.45*60)
    scale = 45./window[np.abs(f-f0)<4e-7].max()
    ax2.plot(f*86400,scale*window,color='k',alpha=0.3)

    ia = pl.axes([0.20,0.675,0.26,0.28])
    mask = f < 0.1
    ia.plot(f[mask]*86400,dlogl[mask],alpha=0.5,color='C0')
    ia.plot(f[mask]*86400,dlogl_nobg[mask],alpha=0.5,color='C1')
    ia.plot(f[mask]*86400,scale*window[mask],color='k',alpha=0.3)
    ia.axis([0,0.05,0,50])
    ia.axvline(1./365,ymin=0.7,color='k',ls='--')
    ia.axvline(1./53.5,ymin=0.7,color='k',ls='--')
    ia.set_yticklabels('')

def make_j0823_power_spectrum(fignum=8):

    data = get_data('j0823',clobber=False)
    ts = data.get_cells(tcell=300,time_series_only=True,
            trim_zero_exposure=False,use_barycenter=True)
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)
    f,window = core.power_spectrum_fft(ts,exp_only=True)

    pl.close(fignum)
    pl.figure(fignum,(4,4)); pl.clf()
    pl.subplots_adjust(hspace=0,bottom=0.16,top=0.98,left=0.18,right=0.96)

    ax1 = pl.subplot(1,1,1)
    mask = f < 1./86400
    ax1.plot(f[mask]*86400,dlogl[mask],alpha=0.5)
    ax1.plot(f[mask]*86400,dlogl_nobg[mask],alpha=0.5)
    ax1.set_xlabel('Frequency (cycles/day)')
    ax1.set_ylabel('Power')
    ax1.axis([0,0.05,0,40])

    ia = pl.axes([0.46,0.54,0.42,0.42])
    ia.plot(f*86400,dlogl,alpha=0.5)
    ia.plot(f*86400,dlogl_nobg,alpha=0.5)
    f0 = 1./(95.45*60)
    #scale = 45./window[np.abs(f-f0)<4e-7].max()
    scale = 45./5000
    ia.plot(f,window*scale,color='k',alpha=0.3)
    ia.axis([0,72,0,40])
    ia.set_yticklabels('')

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
    ax1.set_xlabel('Frequency (cycles/day)')
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
    ax1.set_xlabel('Frequency (cycles/day)')
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

def make_ls5039_power_spectrum(fignum=10):

    data = get_data('ls5039',clobber=False)
    if (data.max_radius is not None) and (data.max_radius < 10):
        data = get_data('ls5039',clobber=True,max_radius=10)

    ts = data.get_cells(tcell=300,time_series_only=True,
            trim_zero_exposure=False,use_barycenter=True)
    f,window = core.power_spectrum_fft(ts,exp_only=True)
    scale = 50./40000
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)

    forb = 2.963145573933919e-06
    fprec = 2.1777777777777778e-07
    freqs = [fprec,forb,2*forb]
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

    ax1 = pl.subplot(1,3,1)
    ax1.set_yscale('log')
    ax1.plot(f*86400,dlogl_nobg,alpha=0.5,color='C0')
    ax1.plot(f*86400,dlogl_nobg2+add_power,alpha=0.5,color='C1')
    ax1.plot(f*86400,scale*window,color='k',alpha=0.3)
    ax1.set_ylabel('Power')
    ax1.axis([-0.1,20,10,2000])

    ia = pl.axes([0.18,0.48,0.15,0.45])
    ia.plot(f*86400,dlogl_nobg,alpha=0.5,color='C0')
    ia.plot(f*86400,dlogl_nobg2+add_power,alpha=0.5,color='C1')
    ia.plot(f*86400,scale*window,color='k',alpha=0.3)
    ia.axis([0,1,0,1500])

    ax2 = pl.subplot(1,3,2)
    #ia = pl.axes([0.30,0.30,0.60,0.60])
    #mask = (f > 14.2/86400) & (f < 16.2/86400)
    ax2.plot(f*86400,dlogl_nobg,alpha=0.5,color='C0')
    ax2.plot(f*86400,dlogl_nobg2+add_power,alpha=0.5,color='C1')
    ax2.plot(f*86400,scale*window,color='k',alpha=0.3)
    ax2.axis([0,1,0,70])
    ax2.set_xlabel('Frequency (cycles/day)')

    ax3 = pl.subplot(1,3,3)
    scale *= 0.6
    ax3.plot(f*86400,dlogl_nobg,alpha=0.5,color='C0')
    ax3.plot(f*86400,dlogl_nobg2,alpha=0.5,color='C1')
    ax3.plot(f*86400,scale*window,color='k',alpha=0.3)
    ax3.plot((f-freqs[1])*86400,scale*window*0.5,color='k',alpha=0.3)
    ax3.plot((f+freqs[1])*86400,scale*window*0.5,color='k',alpha=0.3)
    ax3.plot((f-freqs[2])*86400,scale*window*0.25,color='k',alpha=0.3)
    ax3.plot((f+freqs[2])*86400,scale*window*0.25,color='k',alpha=0.3)
    ax3.axis([14.4,15.8,0,70])
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
    ax2.set_xlabel('Frequency (cycles/day)')

    data = get_data('ls5039',clobber=True,do_pickle=False,max_radius=2)
    ts = data.get_cells(tcell=300,time_series_only=True,
            trim_zero_exposure=False,use_barycenter=True)
    f,window = core.power_spectrum_fft(ts,exp_only=True)
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)

    ax3 = pl.subplot(1,3,3)
    ax3.plot(f*86400,dlogl_nobg,alpha=0.5,color='C0')
    ax3.plot(f*86400,dlogl,alpha=0.5,color='C1')
    ax3.axis([0,1,0,1500])

def make_cygx3_plot(fignum=13):
    p0 = 0.199693736062
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
    ax1.set_ylabel("Relative Flux / Power")

    left_edges = r2[:,0]-r2[:,1]
    right_edges = r2[:,0]+r2[:,1]
    edges = mjd2met(np.append(left_edges,right_edges[-1]))
    scales = r2[:,2]
    ts = data.get_cells(tcell=600,time_series_only=True,trim_zero_exposure=False,scale_series=[edges,scales])
    f,dlogl_nobg,dlogl,dlogl_null = core.power_spectrum_fft(ts)
    fcygx3 = 1./(0.19968476+5.42e-10*(56561-40000))
    freqs = [fcygx3/86400]
    corr,pows = core.get_orbital_modulation(ts,freqs)
    f2,dlogl_nobg2,dlogl2,dlogl_null2 = core.power_spectrum_fft(ts,exposure_correction=corr)
    ts = data.get_cells(tcell=600,time_series_only=True,trim_zero_exposure=False)
    f3,dlogl_nobg3,dlogl3,dlogl_null3 = core.power_spectrum_fft(ts)

    add_power = np.zeros_like(dlogl_nobg2)
    for freq,p in zip(freqs,pows):
        idx = np.argmin(np.abs(f[1:]-freq))
        add_power[idx] = p

    ax2 = pl.subplot(1,3,2)
    ax2.clear()
    ax2.plot(f*86400,dlogl_nobg3,alpha=0.5,color='C0')
    #ax2.plot(f*86400,dlogl_nobg2+add_power,alpha=0.5,color='C1')
    #ax2.plot(f*86400,scale*window,color='k',alpha=0.3)
    ax2.plot(f*86400,dlogl_nobg2+add_power,alpha=0.5,color='C1')
    ax2.set_xlabel('Frequency (cycles/day)')
    #ax2.set_ylabel('Power')
    ax2.axis([-1,36,0,240])
    ax2.axvline(1./p0,ymin=0.9,ymax=1.0,ls='--',color='C3',alpha=0.3)


    f,window = core.power_spectrum_fft(ts,exp_only=True)
    ax3 = pl.subplot(1,3,3)
    ax3.clear()
    ax3.plot(f*86400,dlogl_nobg,alpha=0.5,color='C0')
    ax3.plot(f*86400,dlogl_nobg2+add_power,alpha=0.5,color='C1')
    ax3.plot(f*86400./3,window/14570*100*3,color='k',alpha=0.3)
    #ax3.set_ylabel('Power')

    ax3.set_xlabel('Frequency (cycles/day)')
    #ax3.axis([-1,36,0,220])
    ax3.axis([1./p0-0.05,1./p0+0.05,0,240])
    ax3.axvline(1./p0,ymin=0.9,ymax=1.0,ls='--',color='C3',alpha=0.3)

#ia = pl.axis([0.78,0.48,0.18,0.45])

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



def make_figure_7():
    make_geminga_power_spectrum(fignum=7)
    pl.savefig('fig7.pdf')

def make_figure_8():
    make_j0823_power_spectrum(fignum=8)
    pl.savefig('fig8.pdf')

def make_figure_9():
    make_j1018_power_spectrum(fignum=9)
    pl.savefig('fig9.pdf')
"""
Notes on power spectra:
    Both 1231 and 0633 are pretty clean, showing almost no artifacts, and
    adhering pretty nicely to a chi^2 distribution.

    3C 279 seems to have a great deal of spectral leakage and possibly
    PSF modulation.

    J0823 shows the 91.25 day modulation from Vela's PSF, and it goes away
    with the background reduction method! [PLOT]

    J0835 shows very strong peaks at 1-day, but this goes away using the
    background reduction method.  Precession does not.

    LS 5039 is an interesting one.  A precession peak, smaller in 2dg
    aperture than 5dg (perhaps just a photon argument).  Lots of low
    frequency noise that goes away with bg method.  Is it real?  What's
    nearby?  Additionally, the signal strength drops *a lot* with the
    background method.

    LSI +61 303: similar, except low frequency noise stays (because it
    is real, I reckon).  Also two peaks at orbital frequency, is that the
    "super orbital" period, or just a mixing?

    J1018 is very clean except for orbital period, and the amplitudes are
    similar.
"""

set_rcParams()
