""" Use astropy to quickly convert Fermi METs to TDB at the barycenter.
"""

from astropy import constants as const
from astropy import coordinates as coord
from astropy import time
from astropy import units as u
import numpy as np

def met2mjd(times,mjdref=51910+7.428703703703703e-4):
    times = np.asarray(times,dtype=np.float128)
    return times*(1./86400)+mjdref

def mjd2met(times,mjdref=51910+7.428703703703703e-4):
    times = np.asarray(times,dtype=np.float128)
    return (times-mjdref)*86400

def met2tdb(met,ra,dec):
    """ Return approximate METs corrected to the solar system barycenter.

    This method IGNORES spacecraft position, so it is not intended for
    high precision work.

    It uses the default astropy ephemeris (DE405) which is certainly good
    enough for this.

    Parameters
    ----------
    met : this is Mission Elapse Time, topocentric (i.e. GPS/TT)
    ra : position to barycenter (deg)
    dec : position to barycenter (deg)

    Returns
    -------
    met(tdb) : time(s) corrected to SSB, in TDB, referenced to Fermi epoch
    """
    mjds = met2mjd(met)
    times = time.Time(mjds,format='mjd',scale='tt')
    zero = np.full(len(times),0)
    gcrs = coord.GCRS(zero*u.deg,zero*u.deg,zero*u.m,obstime=times)
    cpos = gcrs.transform_to(coord.ICRS()).cartesian.xyz
    sky = coord.SkyCoord(ra*u.deg,dec*u.deg)
    spos = sky.icrs.represent_as(coord.UnitSphericalRepresentation).represent_as(coord.CartesianRepresentation).xyz
    delay = spos@cpos/const.c # (3,1) x (3,N)
    tdbs = times.tdb + time.TimeDelta(delay,scale='tdb')
    return mjd2met(tdbs.mjd)

def test():
    """ NB this requires v2.0 or earlier of fermitools.
    """
    from skymaps import PythonUtilities
    mjds = np.sort(np.random.rand(10000)*1000+55555)
    mets = mjd2met(mjds)
    import time
    t1 = time.time()
    tdbs_astro = met2tdb(mets,10,10)
    t2 = time.time()
    ft2 = '/tank/kerrm/fermi_data/spacecraf/lat_spacecraft_weekly_w009_p310_v001.fits'
    tdbs_fermi = mets.copy().astype(np.float64)
    t3 = time.time()
    PythonUtilities.met2tdb(tdbs_fermi,10,10,ft2)
    t4 = time.time()
    print(f'astropy code ran in {(t2-t1):.3f}s.')
    print(f'fermist code ran in {(t4-t3):.3f}s.')
    # Just insist on, I dunno, 100mus?
    assert(np.all(np.abs(tdbs_astro-tdbs_fermi)<1e-4))
    return tdbs_astro,tdbs_fermi
