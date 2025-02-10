from collections import deque

from astropy.io import fits
import numpy as np

def Gti(*args,**kwargs):
    """ Factory method to emulate multiple constructors.
    """
    if len(args) == 1:
        # assume this is a file name
        if 'gti_ext' in kwargs.keys():
            pass
        gti_ext = 'gti'
        hdu = fits.open(args[0])[gti_ext]
        t0 = hdu.data['start']
        t1 = hdu.data['stop']

    elif len(args) == 2:
        t0,t1 = args
    return _Gti(t0,t1)

class _Gti():
    """ A pure python replacement for the old skymaps.Gti class."""

    def __init__(self,starts,stops):
        stops = np.atleast_1d(stops)
        a = np.argsort(stops)
        self._t0 = np.atleast_1d(starts)[a]
        self._t1 = np.atleast_1d(stops)[a]
        assert(np.all(self._t0[1:] >= self._t1[:-1]))

    def accept(self,t):
        idx = np.minimum(np.searchsorted(self._t1,t),len(self._t1)-1)
        return (self._t0[idx] <= t) & (self._t1[idx] > t)

    def combine(self,other):
        if other._t1[-1] <= self._t0[0]:
            # other is entirely before us
            self._t0 = np.append(other._t0,self._t0)
            self._t1 = np.append(other._t1,self._t1)
        elif other._t0[0] >= self._t1[-1]:
            # other is entirely after us
            self._t0 = np.append(self._t0,other._t0)
            self._t1 = np.append(self._t1,other._t1)
        else:
            # there is a non-trivial intersection; complete in two steps,
            # truncating any intersecting GTIs such that they don't overlap,
            # then merging any GTIs with identical boundaries
            st0 = np.append(self._t0,np.inf)
            ot0 = np.append(other._t0,np.inf)
            st1 = np.append(self._t1,np.inf)
            ot1 = np.append(other._t1,np.inf)
            i0 = np.searchsorted(ot1,self._t1)
            i1 = np.searchsorted(st1,other._t1)
            # TODO -- need to handle the case of the far edge, I think
            new_st1 = np.minimum(self._t1,ot0[i0])
            new_ot1 = np.minimum(st0[i1],other._t1)
            new_t0 = np.append(self._t0,other._t0)
            new_t1 = np.append(new_st1,new_ot1)
            dt = new_t1-new_t0
            new_t0 = new_t0[dt>0]
            new_t1 = new_t1[dt>0]
            a = np.argsort(new_t0)
            new_t0 = new_t0[a]
            new_t1 = new_t1[a]
            nt0 = [new_t0[0]]
            nt1 = []
            for i in range(1,len(new_t0)-1):
                if new_t0[i] == new_t1[i-1]:
                    continue
                else:
                    nt1.append(new_t1[i-1])
                    nt0.append(new_t0[i])
            nt1.append(new_t1[-1])
            assert(len(nt1)==len(nt0))
            self._t0 = np.asarray(nt0)
            self._t1 = np.asarray(nt1)
        return self

    def intersection(self,other):
        """ Defined via SWIG in the old days to get &=.
        """

        # Search for the shortest set within the longest set
        t0,t1 = self._t0,self._t1
        ot0,ot1 = other._t0,other._t1
        if len(ot0) > len(t0):
            t0,t1,ot0,ot1 = ot0,ot1,t0,t1

        i0 = np.searchsorted(ot1,t0)
        i1 = np.searchsorted(ot1,t1)
        i0 = np.minimum(len(ot1)-1,i0)
        i1 = np.minimum(len(ot1)-1,i1)

        new_t0 = deque()
        new_t1 = deque()

        for i,(ii0,ii1) in enumerate(zip(i0,i1)):
            for j in range(ii0,ii1+1):
                ct0 = max(t0[i],ot0[j])
                ct1 = min(t1[i],ot1[j])
                if ct0 < ct1:
                    new_t0.append(ct0)
                    new_t1.append(ct1)

        self._t0 = np.asarray(new_t0)
        self._t1 = np.asarray(new_t1)
        return self

    def applyTimeRangeCut(self,tmin,tmax):
        """ Return a new GTI with support only within [tmin,tmax)."""
        return Gti(tmin,tmax).intersection(self)

    def equal(self,other):
        if not (len(self._t0) == len(other._t0)):
            return False
        if not np.allclose(self._t0,other._t0):
            return False
        if not np.allclose(self._t1,other._t1):
            return False
        return True

    def computeOntime(self):
        return np.sum(self._t1-self._t0)

    def minValue(self):
        """ Return smallest time in GTI.

        By construction, this is the first element the start times.
        """
        return self._t0[0]

    def maxValue(self):
        """ Return largest time in GTI.

        By construction, this is the last element the stop times.
        """
        return self._t1[-1]

    def get_edges(self,starts=True):
        if starts:
            return self._t0
        return self._t1

def test_Gti(get_test_data=False):
    t0 = [0.0,0.6,1.5,4.5,5.0,7.1,8.3]
    t1 = [0.4,1.0,2.0,4.7,6.0,7.9,8.5]
    ot0 = [-1.0,0.3,2.5,3.8,5.5,7.0,7.3,7.6,8.4]
    ot1 = [-0.5,0.8,3.5,4.8,5.6,7.2,7.4,8.0,8.6]
    intersect_answer_t0 = np.asarray([0.3,0.6,4.5,5.5,7.1,7.3,7.6,8.4])
    intersect_answer_t1 = np.asarray([0.4,0.8,4.7,5.6,7.2,7.4,7.9,8.5])
    union_answer_t0 = np.asarray([-1.0,0.0,1.5,2.5,3.8,5.0,7.0,8.3])
    union_answer_t1 = np.asarray([-0.5,1.0,2.0,3.5,4.8,6.0,8.0,8.6])
    if get_test_data:
        return t0,t1,ot0,ot1,intersect_answer_t0,intersect_answer_t1,union_answer_t0,union_answer_t1
    g1 = Gti(t0,t1)
    g2 = Gti(ot0,ot1)
    assert(np.allclose(g1.computeOntime(),3.5))
    g1.intersection(g2)
    assert(np.all(g1._t0==intersect_answer_t0))
    assert(np.all(g1._t1==intersect_answer_t1))
    g2.intersection(g1)
    assert(np.all(g2._t0==intersect_answer_t0))
    assert(np.all(g2._t1==intersect_answer_t1))
    assert(np.allclose(g1.computeOntime(),1.2))
    assert(np.allclose(g2.computeOntime(),1.2))
    g1 = Gti(t0,t1)
    test_accept = [-0.5,0.2,0.5,0.6,1.0,1.3,10]
    accept_answer = np.asarray([0,1,0,1,0,0,0],dtype=bool)
    assert(np.all(g1.accept(test_accept)==accept_answer))
    g3 = g1.applyTimeRangeCut(0,10)
    assert(g1.equal(g3))
    g3 = g1.applyTimeRangeCut(0.2,8.0)
    answer_t0 = np.asarray([0.2, 0.6, 1.5, 4.5, 5. , 7.1])
    answer_t1 = np.asarray([0.4, 1. , 2. , 4.7, 6. , 7.9])
    assert(np.all(g3._t0==answer_t0))
    assert(np.all(g3._t1==answer_t1))
    assert(np.allclose(g3.computeOntime(),3.1))

    g1 = Gti(t0,t1)
    g2 = Gti(ot0,ot1)
    g1.combine(g2)
    assert(np.all(g1._t0==union_answer_t0))
    assert(np.all(g1._t1==union_answer_t1))


    try:
        from skymaps import Gti as cGti
        cg1 = cGti(t0,t1)
        cg2 = cGti(ot0,ot1)
        cg1.intersection(cg2)
        assert(np.allclose(g2._t0,cg1.get_edges(True)))
        assert(np.allclose(g2.get_edges(True),cg1.get_edges(True)))
        assert(np.allclose(g2._t1,cg1.get_edges(False)))
        assert(np.allclose(g2.get_edges(False),cg1.get_edges(False)))
    except ImportError:
        print('Could not complete comparison against C++ implementation.')
