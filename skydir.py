""" Implement a drop-in SkyDir replacement using astropy.
"""
from astropy import coordinates

EQUATORIAL = 1
GALACTIC = 0

class SkyDir():

    def __init__(self,lon,lat,frame=EQUATORIAL):
        if frame == EQUATORIAL:
            self._coord_eq = coordinates.SkyCoord(lon,lat,unit="deg",frame=coordinates.ICRS)
            self._coord_lb = self._coord_eq.transform_to(coordinates.Galactic)
        elif frame == GALACTIC:
            self._coord_lb = coordinates.SkyCoord(lon,lat,unit="deg",frame=coordinates.Galactic)
            self._coord_eq = self._coord_lb.transform_to(coordinates.ICRS)
        else:
            raise ValueError('Frame not supported.')
        self._frame = frame

    def ra(self):
        return self._coord_eq.ra.deg

    def dec(self):
        return self._coord_eq.dec.deg

    def l(self):
        return self._coord_lb.l.deg
            
    def b(self):
        return self._coord_lb.b.deg
        
