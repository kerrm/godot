import core
import cPickle
from astropy.io import fits

data_path = '/data/kerrm/photon_data'

def get_data(source,clobber=False,do_pickle=True,**data_kwargs):

    if 'use_phi' not in data_kwargs:
        data_kwargs['use_phi'] = True

    if not clobber:
        try:
            data = cPickle.load(file('%s_data.pickle'%source))
            print 'returning cached version of Data object'
            return data
        except:
            pass

    data = None

    if source == 'j1018':
        jname = 'J1018-5856'

    elif source.startswith('j1231'):
        jname = 'J1231-1411'

    elif source.startswith('j2021'):
        jname = 'J2021+4026'

    elif source.startswith('j1311'):
        jname = 'J1311-3430'

    elif source.startswith('j2241'):
        jname = 'J2241-5236'

    elif source.startswith('j2032'):
        jname = 'J2032+4127'

    elif source.startswith('cygx3'):
        jname = 'J2032+4057_fake'

        #spectrum = lambda E: (E/1000)**-2.1

    elif source.startswith('j0633'):
        jname = 'J0633+1746'

    elif source.startswith('j0823'):
        jname = 'J0823.3-4205c_fake'

    elif source.startswith('j0534'):
        jname = 'J0534+2200'

    elif source == 'eridani':
        ra = 53.2327
        dec = -9.45826
        ft1files = sorted(glob.glob('/data/kerrm/eps_eridani/gtsrcprob*.fits'))
        ft2files = ['data/tyrel_ft2.fits']
        spectrum = lambda E: (E/1000)**-3

        data = core.Data(ft1files,ft2files,ra,dec,'Eridani',
                base_spectrum=spectrum,zenith_cut=90,
                use_weights_for_exposure=exposure_weights,use_phi=use_phi,
                max_radius=15)

    elif source == 'j0835_topo':
        jname = 'J0835-4510'

    elif source == 'lsi':
        jname = 'J0240+6113_fake'

    elif source == 'ls5039':
        jname = 'J1826-1450_fake'

        spectrum = lambda E: (E/1000)**-2.1

    elif source == '3c279':
        jname = 'J1256-0547_fake'

    else:
        print 'Did not recognize a source alias.  Assuming source==jname.'
        jname = source
        #raise NotImplementedError('did not recognize %s'%source)

    if (data is not None):
        if do_pickle:
            cPickle.dump(data,file('%s_data.pickle'%source,'wb'),protocol=2)
        return data

    spectrum = lambda E: (E/1000)**-2.1
    ft1files = ['%s/%s_%s.fits'%(data_path,jname,'bary' if 'bary' in source else 'topo')]

    # load position from FT1 file
    f = fits.open(ft1files[0])
    try:
        ra,dec,rad = f[1]._header['DSVAL2'].lstrip('CIRCLE(').rstrip(')').split(',')
    except:
        ra,dec,rad = f[1]._header['DSVAL1'].lstrip('CIRCLE(').rstrip(')').split(',')

    print 'Using ra = %s, dec = %s, with extraction radius %s'%(ra,dec,rad)
    ra = float(ra)
    dec = float(dec)
    f.close()

    if ('max_radius' in data_kwargs) and (float(rad) < data_kwargs['max_radius']):
        print 'Warning, specified max_radius=%s but data cuts imply %s.'%(data_kwargs['max_radius'],rad)

    ft2files = ['%s/ft2.fits'%data_path]

    data = core.Data(ft1files,ft2files,ra,dec,'PSR%s'%jname,
            base_spectrum=spectrum,zenith_cut=100,**data_kwargs)
    if do_pickle:
        cPickle.dump(data,file('%s_data.pickle'%source,'wb'),protocol=2)
    return data


