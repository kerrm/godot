""" Some routines for manipulating Fermi event (FT1) files and their data.
"""

from astropy.io import fits
from astropy.io.fits.column import _ColumnFormat
import numpy as np

def parse_dss(hdr):
    """ Return some information about the data cuts in an FT1 file using
    the DSS keywords.

    Specifically, the relevant max radius, energy, and zenith angle cuts.

    Parameters
    ----------
    The header of the EVENTS HDU for the FT1 file.

    Returns
    -------
    (ra,dec,maxrad), (emin,emax), (zmax), (evtclass, recon)
    """
    rcuts = None
    ecuts = None
    zmax = None
    evtclass = None
    ndss = len([x for x in hdr.keys() if x.startswith('DSTYP')])
    for idss in range(1,ndss+1):
        dstyp = hdr[f'DSTYP{idss}'].strip()
        dsval = hdr[f'DSVAL{idss}'].strip()
        if dstyp.startswith('BIT_MASK'):
            toks = dstyp.lstrip('BIT_MASK(').rstrip(')').split(',')
            if toks[0] == 'EVENT_CLASS':
                evtclass = (int(toks[1]),toks[2])
            continue
        if dsval.startswith('CIRCLE'):
            rcuts = [float(x) for x in dsval[8:-1].split(',')]
            continue
        if dstyp == 'ZENITH_ANGLE':
            zmax = float(dsval.split(':')[-1])
            continue
        if dstyp == 'ENERGY':
            ecuts = [float(x) for x in dsval.split(':')]
            continue
    return rcuts,ecuts,zmax,evtclass

def load_bitfield(fname,field_name,hdu_name=1):
    """ Use astropy to load a bitfield.  This circumvents the standard
        approach, which will unpack the bits into a 2D numpy bool array,
        which can be quite large and makes things very slow.

    NB this method relies on lazy evaluation, so for "safety", open the
    file afresh.
    """
    with fits.open(fname) as f:
        hdu = f[hdu_name]
        # change from 32X (or whatver) to unsigned integer
        c = hdu.columns[field_name]
        old_format = c.format
        c.format = _ColumnFormat('J')
        dat = hdu.data[field_name].copy() # is the copy needed?
        c.format = old_format
        return dat

def event_type_to_psf_type(et):
    """ Convert the integer array (e.g. from load_bitfield) of EVENT_TYPE
        to the canonical PSF type labels (0-3).
    """
    rvals = np.full(len(et),-1,dtype=np.int8)
    for i in range(4):
        mask = (et & 2**(i+2)) > 0
        rvals[mask] = i
    assert(not np.any(rvals < 0))
    return rvals

def event_type_to_fb_type(et):
    """ Convert the integer array (e.g. from load_bitfield) of EVENT_TYPE
        to the canonical FRONT (0)/BACK (1) type.
    """
    types = np.full(len(et),-1,dtype=np.int8)
    for i in range(2):
        mask = (et & (2**(i+0))) > 0
        types[mask] = i
    assert(not np.any(types < 0))
    return types

def load_psf_type(hdu):
    """ Load the EVENT_TYPE bitmask and convert it into an integer array
        whose entries are the PSF types (0-3).
    """
    dat = load_bitfield(hdu,'EVENT_TYPE')
    return event_type_to_psf_type(dat)

def radius_cut(hdu,ra0,dec0,max_radius):
    """ Return a mask limiting events to given radius around given ra/dec.

    Parameters
    ----------
    hdu : the EVENTS HDU of an FT1 file
    ra0 : center RA in deg
    dec0 : center Decl. in deg
    max_radius : maximum radius in degrees
    """
    ra = np.radians(hdu.data['ra'])
    dec = np.radians(hdu.data['dec'])
    cdec,sdec = np.cos(dec),np.sin(dec)
    cdec_src = np.cos(np.radians(dec0))
    sdec_src = np.sin(np.radians(dec0))
    # cosine(polar angle) of source in S/C system
    cosines  = cdec_src*np.cos(dec)*np.cos(ra-np.radians(ra0)) + sdec_src*np.sin(dec)
    mask = (cosines >= np.cos(np.radians(max_radius)))
    return mask
