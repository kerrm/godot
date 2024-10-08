import numpy as np

class PiecewiseScaler(object):

    def __init__(self,starts,stops,scales):
        """
        Parameters
        ----------
        starts : beginning of scale intervals (MET)
        stops : end of scale intervals (MET)
        scales : flux multiplier for each interval
        """

        def _add_ends(a,v0,v1):
            rvals = np.empty(len(a)+2)
            rvals[0] = v0
            rvals[-1] = v1
            rvals[1:-1] = a
            return rvals

        # add on a buffer to return the final value at either end
        self._starts = _add_ends(starts,0,stops[-1])
        self._stops = _add_ends(stops,starts[0],np.inf)
        self._scales = _add_ends(scales,scales[0],scales[-1])
        self._tbin = self._stops-self._starts
        x = self._tbin.astype(np.float128)
        y = self._scales.astype(np.float128)
        self._prod = (x*y).astype(np.float64)
        self._cprod = np.cumsum(x*y).astype(np.float64)

    def __call__(self,met_start,met_stop=None):
        """ Match the indicated exposure intervals, tabulated by the
        arguments, and intersect them with the variability intervals.
        Any exposure intervals which are completely containined in a 
        variability interval simply receive that flux; otherwise, the 
        integral over the multiple spanned intervals is done.

        Parameters
        ----------
        met_start : ndarray of start times
        met_stop : ndarray of stop times; if None, then the times are
            assumed to be "instantaneous", i.e. event times, and will do
            a simple lookup.
        """
        T0 = np.atleast_1d(met_start)
        idx0 = np.searchsorted(self._stops,T0)
        if met_stop is None:
            # simple lookup
            mask = idx0 >= len(self._scales)
            return self._scales[idx0]
        T1 = np.atleast_1d(met_stop)
        idx1 = np.searchsorted(self._stops,T1)
        # Edge case: when the the intervals are aligned, the start will be
        # associated with the previous interval, so for that case, bump
        # up the index.
        idx0[T0==self._stops[idx0]] += 1
        # Identify the inputs which span more than one interval
        mask = idx0 != idx1
        if not np.any(mask):
            return self._scales[idx0]

        # This is the end of the first variability interval which overlaps
        # the exposure interval.
        t0 = self._stops[idx0][mask]
        # This is the start of the last variability inteval which overlaps
        # the exposure interval.
        t1 = self._starts[idx1][mask] 
        # And the corresponding fluxes.
        F0 = self._scales[idx0][mask]
        F1 = self._scales[idx1][mask]
        assert(np.all(t0>T0[mask]))
        assert(np.all(T1[mask]>t1))
        edges = F1*(T1[mask]-t1) + F0*(t0-T0[mask])
        #addon = np.asarray([np.sum(self._tbin[i0+1:i1]*self._scales[i0+1:i1]) for i0,i1 in zip(idx0[mask],idx1[mask])])
        i0 = idx0[mask]+1
        i1 = idx1[mask]
        addon2 = np.zeros(len(i0))
        a = np.flatnonzero((i1-i0)==1)
        addon2[a] = self._prod[i0[a]]
        a = np.flatnonzero((i1-i0)>1)
        addon2[a] = self._cprod[i1[a]]-self._cprod[i0[a]]
        rvals = np.empty(len(mask),dtype=float)
        rvals[mask] = (edges + addon2)/(T1[mask]-T0[mask])
        rvals[~mask] = self._scales[idx0[~mask]]
        return rvals
