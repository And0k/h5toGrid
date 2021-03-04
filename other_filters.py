import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pywt

from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
from numba import njit, objmode, typed, types

if __debug__:
    import matplotlib
    from matplotlib import pyplot as plt

# l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
l = logging.getLogger(__name__)

dt64_1s = np.int64(1e9)

#@njit: Invalid use of <function mean>) with argument "axis"
def mad(data, axis=None):
    """Instead this can use: from statsmodels.robust import mad  # or use pd.DataFrame().mad()"""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

# from scipy.signal import find_peaks_cwt
# -----------------------------------------------------------------
@njit
def rep2mean(y, bOk=None, x=None):
    """
    Replaces bad indexes in y using numpy.interp()
    :param y:
    :param bOk: logical not of bad indexes
    :param x: same length as y or None
    :return:
    """
    if bOk is None:
        b = np.logical_not(np.isnan(y))  # numba KeyError: 'Failed in nopython mode pipeline (step: ensure IR is legal prior to lowering)\n"more than one definition for \'bOk\'"'
    else:
        b = bOk
    try:
        if x is None:
            x_bad = np.arange(len(y), dtype=np.float_)
            x_ok = np.flatnonzero(b.astype(np.float_))
            y_ok = np.extract(b, y)  # y[bOk]
            return np.interp(x_bad, x_ok, y_ok)
        else:
            bBad = np.logical_not(b)
            y[bBad] = np.interp(x[bBad], x[b], y[b])
            return y
    except Exception:  # ValueError as e:  # array of sample points is empty - replaced for numba
        #l.exception('rep2mean error')
        b = np.isfinite(y)
        if bOk is not None:
            b = np.logical_and(b, bOk)
        if b.sum() == 0:
            print('rep2mean can not replace NaNs')
            return y + np.NaN  # (np.NaN if bOk is None else
        else:  # may be 1 good point? - return constant
            print('rep2mean: strange condition on Exception!')
            return np.where(bOk if bOk is not None else b, y[b], np.NaN)
    # def rep2mean(x, bOk):
    # old= lambda x, bOk: np.interp(np.arange(len(x)), np.flatnonzero(bOk), x[bOk], np.NaN,np.NaN)

#    x[~bOk]= np.interp(np.arange(len(x)), np.flatnonzero(bOk), x[bOk], np.NaN,np.NaN)

@njit
def b1spike(a, max_spike=0):
    """
    Single spike detection
    Note: change of a at edge bigger than max_spike is treated as spike too
    :param a:
    :param max_spike:
    :return: bool array of where is spike in a
    """
    diff_x = np.diff(a)
    b_single_spike_1 = lambda bad_u, bad_d: np.logical_or(
        np.logical_and(np.append(bad_d, True), np.append(True, bad_u)),  # spike to down
        np.logical_and(np.append(bad_u, True), np.append(True, bad_d)))  # spike up

    return b_single_spike_1(diff_x < -max_spike, diff_x > max_spike)
    # to do: try bOk[-1] = True; bOk = np.logical_or(bOk, np.roll(bOk, 1))

@njit
def b1spike_up(a, max_spike):

    diff_x = np.diff(a)
    b_single_spike_1 = lambda bad_u, bad_d: \
        np.logical_and(np.append(bad_u, True), np.append(True, bad_d))  # spike up
    return b_single_spike_1(diff_x < -max_spike, diff_x > max_spike)

#@njit: "searchsorted with argument(s) ... ValueError: Invalid value given for 'side': unicode_type"
def i_move2good(GoodIndIn, bad, side='left'):
    """
    moves indices to keep its number and relative position when removed masked data
    :param GoodIndIn: indexes of some data in array X
    :param bad:      mask to delete elements from X
    :return:          indexes of some data after deletion takes place
                                    0 1 2 3 4 5 6
    move2GoodI([0, 1, 5], np.array([0,0,0,0,0,1,0]))
    >>> [0,1,5]
    move2GoodI([0, 1, 5], np.array([0,0,0,0,0,0,1]), 'right')
    >>> [0,1,5]

    move2GoodI([0, 1, 5], np.array([0,0,0,0,1,0,0]))
    >>> [0,1,4]
    move2GoodI([0, 1, 5], np.array([0,0,0,0,0,1,0]), 'right')
    >>> [0,1,4]


    move2GoodI([0, 1, 5], np.array([0,0,0,0,0,0,1]), 'right')
    >>> [0,1,5]
    """
    ind = np.arange(bad.size)[np.logical_not(bad)]  # logical_not will convert ints to bool array
    ind_out = np.searchsorted(ind, GoodIndIn, side=side)
    if side == 'right':
        ind_out -= 1
    # np.clip(ind_out, 0, ind.size - 1, out=ind_out)
    return ind_out
    # Same porpose code which mostly works the same but can return negative indexes:
    # s= np.int32(np.cumsum(bBad))
    # ind_out= np.int32(GoodIndIn) - s[GoodIndIn]
    # ind= np.flatnonzero(bBad[GoodIndIn]) # if some GoodIndIn in bBad regions
    # for k in ind:
    #     f= np.flatnonzero(~bBad[GoodIndIn[k]:])[0]
    #     if f: # data masked -
    #         f += (GoodIndIn[k] - 1)
    #         ind_out[k]= f - s[f]
    #     else:
    #         f= np.flatnonzero(~bBad[:GoodIndIn[k]])[-1]
    #         if f:
    #             ind_out[k]= f - s[f]

@njit
def contiguous_regions(b_ok):
    """
    Finds contiguous True regions of the boolean array "b_ok"
    :param b_ok: 
    :return: 2D array where the first column is the start index of the region and the
    second column is the end index: [[start[0], end[0]] ... [start[-1], end[-1]]].
    If last element of b_ok is True then end[-1] is equal to len(b_ok)
    """
    d = np.ediff1d(np.int8(b_ok), to_begin=b_ok[0], to_end=b_ok[-1])  # b_ok.astype(int)
    return np.flatnonzero(d).reshape((-1, 2))

@njit
def find_smallest_elem_as_big_as(seq: np.ndarray, subseq: typed.List, elem) -> int:
    """
    Returns the index of the smallest element in subsequence as big as seq[elem].

    seq[elem] must not be larger than every element in subsequence.
    The elements in subseq are indices in seq.
    Uses binary search.
    """

    low = 0
    high = len(subseq) - 1

    while high > low:
        mid = (high + low) // 2
        # If the current element is not as big as elem, throw out the low half of sequence.
        if seq[subseq[mid]] < seq[elem]:
            low = mid + 1
        else:  # If the current element is as big as elem, throw out everything bigger, but
            # keep the current element.
            high = mid

    return high

@njit
def longest_increasing_subsequence_i(seq: np.ndarray) -> List[Any]:
    """
    Finds the longest increasing subseq in sequence using dynamic
    programming and binary search (per http://en.wikipedia.org/wiki/Longest_increasing_subsequence).
    :param seq: sequence of supported type for numba
    :return: subseq indexes of seq

    This is O(n log n) optimized_dynamic_programming_solution from
    https://stackoverflow.com/questions/2631726/how-to-determine-the-longest-increasing-subsequence-using-dynamic-programming/36836767#36836767
    """
    result = typed.List()  # np.float64
    l = len(seq)
    # if ~l:  # empty sequence of same type as input:
    #     return result  # NOT WORKS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! HELP

    # Both of these lists hold the indices of elements in sequence and not the
    # elements themselves.

    # This list will always be sorted.
    smallest_end_to_subseq_of_length = typed.List()

    # This array goes along with sequence (not smallest_end_to_subseq_of_length).
    # Following the corresponding element in this array repeatedly will generate
    # the desired subsequence.
    parent = typed.List([-1] * l)  #-np.ones(l, np.int32)  # [None] * l

    for elem in range(l):
        # We're iterating through sequence in order, so if elem is bigger than the
        # end of longest current subsequence, we have a new longest increasing subsequence.
        if (len(smallest_end_to_subseq_of_length) == 0 or
                seq[elem] > seq[smallest_end_to_subseq_of_length[-1]]):
            # If we are adding the first element, it has no parent.  Otherwise, we
            # need to update the parent to be the previous biggest element.
            if len(smallest_end_to_subseq_of_length) > 0:
                parent[elem] = smallest_end_to_subseq_of_length[-1]
            smallest_end_to_subseq_of_length.append(elem)
        else:
            # If we can't make a longer subsequence, we might be able to make a
            # subsequence of equal size to one of our earlier subsequences with a
            # smaller ending number (which makes it easier to find a later number that
            # is increasing).
            # Thus, we look for the smallest element in
            # smallest_end_to_subsequence_of_length that is at least as big as elem
            # and replace it with elem.
            # This preserves correctness because if there is a subsequence of length n
            # that ends with a number smaller than elem, we could add elem on to the
            # end of that subsequence to get a subsequence of length n+1.
            location_to_replace = find_smallest_elem_as_big_as(seq, smallest_end_to_subseq_of_length, elem)
            smallest_end_to_subseq_of_length[location_to_replace] = elem
            # If we're replacing the first element, we don't need to update its parent
            # because a subsequence of length 1 has no parent.  Otherwise, its parent
            # is the subsequence one shorter, which we just added onto.
            if location_to_replace != 0:
                parent[elem] = smallest_end_to_subseq_of_length[location_to_replace - 1]

    # Generate the longest increasing subsequence by backtracking through parent.

    icur_parent = smallest_end_to_subseq_of_length[-1]


    if False:  # if need its values
        while icur_parent != -1:  # is not None:
            result.append(seq[icur_parent])
            icur_parent = parent[icur_parent]
    else:  # if need its indices
        while icur_parent != -1:
            result.append(icur_parent)
            icur_parent = parent[icur_parent]

    result.reverse()
    return result


# # variant 2
# def longest_increasing_subsequence_i(seq):
#     """
#
#     :param seq:
#     :return:
#
#     Returns longest subsequence (non-contiguous) of seq that is
#     strictly non decreasing.
#     """
#     from math import ceil
#
#     n = len(seq)
#     p = np.empty(n)
#     m = np.empty(n + 1)
#
#     l = 0
#     for i in range (n-1):
#         # Binary search for the largest positive j ≤ L
#         # such that X[M[j]] < X[i]
#         lo = 1
#         hi = l
#         while lo <= hi:
#             mid = ceil((lo+hi)/2)
#             if seq[m[mid]] < x[i]:
#                 lo = mid+1
#             else:
#                 hi = mid-1
#
#         # After searching, lo is 1 greater than the
#         # length of the longest prefix of X[i]
#         newL = lo
#
#         # The predecessor of X[i] is the last index of
#         # the subsequence of length newL-1
#         p[i] = m[newL-1]
#         m[newL] = i
#
#         if newL > l:
#             # If we found a subsequence longer than any we've
#             # found yet, update L
#             l = newL
#
#     # Reconstruct the longest increasing subsequence
#     s = np.empty_like(l)
#     k = m[l]
#     for i in range(l-1,0,-1):
#         s[i] = seq[k]
#         k = p[k]
#
#     return s


# from functools import lru_cache
#
# #@lru_cache(maxsize=None)
# def lcs_len(x, y):
#     """
#     finding the Longest Common Subsequence (LCS) of the original string & the sorted string
#     :param x:
#     :param y:
#     :return: (length of seq, seq)
#     """
#     if not x or not y:
#         return 0, []
#
#     xhead, xtail = x[0], x[1:]
#     yhead, ytail = y[0], y[1:]
#     if xhead == yhead:
#         l1, seq1 = lcs_len(xtail, ytail)
#         return l1+1, xhead.append(seq1)
#
#     l1, seq1 = lcs_len(x, ytail)
#     l2, seq2 = lcs_len(xtail, y)
#     if l1 >= l2:
#         return l1, seq1
#     else:
#         return l2, seq2
#
# def longest_increasing_subsequence_i(seq):
#     """
#     Returns longest subsequence (non-contiguous) of seq that is
#     strictly non decreasing.
#     """
#     if isinstance(seq, np.ndarray):
#         seq = seq.tolist()
#
#     #lcs_len.cache_clear()
#     l1, subseq = lcs_len(seq, sorted(seq))
#     l.info('{} need to be deleted', len(seq) - len(subseq))
#     return np.array(subseq)
#
#

@njit
def longest_increasing_subsequence_i___worked_very_long(seq):
    """
    Returns indexes of the longest subsequence (non-contiguous) of seq that is
    strictly non decreasing.
    :param seq:
    :return: np.int32 array
    """
    if not len(seq): return np.int32([])  # seq

    head = [0]  # end position of subsequence with given length

    predecessor = [None]  # penultimate element of l.i.s. ending at given position

    for i in range(1, len(seq)):
        # seq[i] can extend a subsequence that ends with a smaller element
        j = np.searchsorted(seq[np.int32(head)], seq[i])

        # update existing subsequence of length j or extend the longest
        try:
            head[j] = i
        except:
            head.append(i)
        # remember element before seq[i] in the subsequence
        predecessor.append(head[j - 1] if j > 0 else None)

    # return indices ..., p(p(p(i))), p(p(i)), p(i), i
    """
    This gives RecursionError: maximum recursion depth exceeded. Fatal Python error: Cannot recover from stack overflow.
    def trace(i):
        if i is not None:
            yield from trace(predecessor[i])
            yield i
    return np.fromiter(trace(head[-1]), np.int32)
    """

    ## trace subsequence back to output
    result = np.empty_like(head)
    trace_idx = head[-1]
    k = 0
    for k in range(len(predecessor)):
        if trace_idx is None:
            break
        result[k] = trace_idx  # seq[]
        trace_idx = predecessor[trace_idx]

    return result[k::-1]

@njit
def nondecreasing_b(t, longest_increasing_i):
    """
    Find longest nondecreasing subsequence based on longest increasing subsequence:
    fill same values with True

    :param t: array - input data
    :param longest_increasing_i: indexes or bool array, longest increasing subsequence in t
    :return: boolean mask of nondecreasing elements
    """
    print('finding longest nondecreasing subsequence')  #l.debug()
    b_same = np.ediff1d(t, to_end=np.diff(t[-2:])) == 0
    st_en_same = contiguous_regions(b_same)
    # review only found increasing regions:
    # longest_increasing_i can be only at last element in each contigous region so check them
    bok = np.zeros_like(b_same, np.bool_)
    bok[longest_increasing_i] = True
    try:
        st_en_same = st_en_same[bok[st_en_same[:, 1]]]
    except Exception:  # replaced IndexError for @njit compliance:
        # need deal with last element: repeat with decresed end of last region:
        st_en_same[-1, 1] -= 1
        st_en_same = st_en_same[bok[st_en_same[:, 1]]]
        # icrease back:
        st_en_same[-1, 1] += 1
    # fill found regions
    for start, stop in st_en_same:
        bok[start:stop] = True
    return bok

@njit
def repeated2increased(t: np.ndarray, freq: float, b_increased: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Increases time resolution of only repeated or b_increased elements using constant frequency
    Note: can produce inversions where number of repeated elements bigger than freq!
    :param t: numpy.int64 view of np.dtype('datetime64[ns]'), time to correct
    :param freq:
    :param b_increased: numpy.bool array, mask of the elements not requiring a change (adding time) if None then
    calculated using ``append(diff(t) != 0, True)`` - else provide result of this operation if you already have it
    :return: numpy.int64 ndarray
    """
    # uses global dt64_1s = np.int64(1e9)


    # Trend addition corresponded to freq

    # njit-compatible and slightly less precise version of
    # np.linspace(0, len(t) * dt64_1s / freq, len(t), endpoint=False, ...):
    step = dt64_1s / freq
    t_add = np.arange(0, stop=len(t) * step - 1, step=step, dtype=np.int64)
    # indexes of changes + edges
    if b_increased is None:
        # b_inc = np.ediff1d(t, to_begin=1, to_end=1).astype(np.bool_)
        i_inc_e = np.flatnonzero(np.ediff1d(t, to_begin=1, to_end=1))
    else:
        # np.append(True, b_increased
        i_inc_e = np.append(0, np.flatnonzero(b_increased) + 1)

    n_inc_all = len(i_inc_e) - 2
    if n_inc_all > 0:
        i_inc = i_inc_e[:-1]      # indexes of changed elements considering the 1st is changing too
        if __debug__:
            # with objmode():
            #     l.debug(%gHz)
            print('Increasing time resolution of only repeated elements (', len(t) - n_inc_all,
                  ') using constant frequency f =', freq, 'Hz')  # l.debug(%gHz)

        # decrease trend for bad intervals
        n_rep = np.diff(i_inc_e)  # number of repeated elements + 1 (i.e. lengths of intervals with repeated values)
        steps_calc = np.diff(t[i_inc]) / n_rep[:-1]
        if n_rep[-1] > freq:  # the check of lower bound for the last interval is possible only
            step_calc_last = n_rep[-1]/freq  # increase to lower bound
            steps_calc = np.append(steps_calc, step_calc_last)
        else:
            n_rep = n_rep[:-1]

        i_bad_freqs = np.flatnonzero((n_rep > 1) & (step > steps_calc))
        # or n_rep > freq if all changes are minimal
        n_bad_freqs = len(i_bad_freqs)
        if n_bad_freqs:
            print('- failed for', n_bad_freqs, 'intervals of repeated time. Increasing frequency there')
            for i, n, step_calc in zip(i_inc[i_bad_freqs], n_rep[i_bad_freqs], steps_calc[i_bad_freqs]):
                t_add[i:i+n] = np.arange(t_add[i], stop=t_add[i] + step_calc * n - 1, step=step_calc, dtype=np.int64)
                # "- 1" needed if rounding errors to prevent that stop became bigger than start + step * n

        # subtract our trend addition back from elements that initially have been increased
        t_sub = np.zeros_like(t_add)
        t_sub[i_inc] = np.ediff1d(t_add[i_inc], to_begin=0)
        t_add -= np.cumsum(t_sub)
    else:
        print('All elements (', len(t),
              ') are the same! - Increasing them using constant frequency f =', freq, 'Hz')  # l.debug(%gHz)
    return t + t_add



@njit
def rep2mean_with_const_freq_ends(y, b_ok, freq):
    b_bad = ~b_ok
    x = np.flatnonzero(b_bad)
    xp = np.flatnonzero(b_ok)
    y[x] = np.interp(x, xp, y[xp])
    # interp keeps same values here:
    b_edges = (x < xp[0]) | (x > xp[-1])
    if b_edges.any():
        b_bad[:] = True  # reusing variable. Meaning is inversed
        b_bad[x[b_edges]] = False
        y = repeated2increased(y, freq, b_bad)
    return y

#@njit
def make_linear(tim: np.int64, freq: float, dt_big_hole=None) -> bool:
    """
    Corrects tim values (implicitly) by make them linear increasing but excluding big holes

    :param: tim - time with some deviated frequency
    :param: freq - frequency [Hz] for output
    :param: dt_big_hole - min time interval [pd.Timedelta] that will be kept, that is time precision.
        Default: min(2s, 2s/freq)
    :return: True
        tim will be corrected time with constant frequency between "holes"
    If new sequence last time shifts forward in time at dt > dt_big_hole*2 then decrease freq or this interval.
    prints '`' when after filling interval with constant frequency last element shifted less than hole*2, and '.' for big errors detected.

    """
    # uses global dt64_1s = np.int64(1e9)
    dt0 = np.int64(dt64_1s / freq)  # int64, delta time of 1 count
    # Convert/set time values in int64 units
    if dt_big_hole is None:
        dt_big_hole = np.int64(float(2 * dt64_1s) / min(1, freq))
    else:
        dt_big_hole = np.int64(dt_big_hole.total_seconds()) * dt64_1s

    if __debug__:
        print('Linearise time using reference frequency', freq, '(between holes >', round(dt_big_hole/dt64_1s, 2), 's)')

    # find big holes and add at start and end
    d = np.ediff1d(tim, to_begin=dt_big_hole + 1, to_end=dt_big_hole + 1)  # how value changed
    i_st_hole = np.flatnonzero(d > dt_big_hole)  # starts of big holes
    n_holes = i_st_hole.size - 2
    if n_holes:
        i_st_maxhole = i_st_hole[np.argmax(d[i_st_hole])]
        with objmode:
            l.warning('{} holes > {}s in time found: rows {}{}! Max hole={}s at {}'.format(
                n_holes, dt_big_hole / dt64_1s,
                i_st_hole[1:(1 + min(n_holes, 10))],
                '' if n_holes > 10 else '... ',
                         d[i_st_maxhole] / dt64_1s, i_st_maxhole)
                )

    # i_st_hole[ 0]= -1
    # i_st_hole[-1]-= 1
    #@njit  # Untyped global name 'njit': cannot determine Numba type of <class 'function'>
    def i_j_yi_yj_yec_yee(i, j, y, dt0=dt0):
        """
        Some values for processing each hole in cycle of for_tim(): i,j,y[i],y[j-1],aproximated_y[j-1] using dt, y[j-1] - aproximated_y[j-1]
        :param i: current index
        :param j: last index + 1 (i.e. start of next interval), useful as it will be used for end of slice
        :param y: time
        :param dt0: int64, delta time of 1 count
        :return: (i, j, y[i], y[j], yend_clc, yend_err) where:
            yend_clc: calculated last time
            yend_err: difference between calculated and given last time = (yj - yend_clc)
        """
        yi = y[i]  # cur  y
        yj = y[j - 1]  # last y
        i = np.int64(i)  # need because of errors on next operation if type is int32
        j = np.int64(j)  # and to prevent possible similar errors when will be operations with outputs
        yend_clc = yi + (j - 1 - i) * dt0  # time expected in last point
        yend_err = yj - yend_clc  # error in time
        # max limit for np.arange, so make it a little bigger than end of result array:
        yend_clc += np.int64(dt0 / 2)
        return (i, j, yi, yj, yend_clc, yend_err)

    # def gen_i_j_yi_yj_yec_yee(i_st_hole):
    #     for i,j in zip(i_st_hole[:-1], i_st_hole[1:]):
    #         yield i_j_yi_yj_yec_yee(i, j, t)
    #     #return (i, j, yi, yj, yend_clc, yend_err)
    #@njit
    def for_tim(i_st_h, t, b_allow_shift_back=True, dt0=dt0):
        """
        Holes processing in ``t``. Implicitly modifies t
        :param i_st_h:
        :param t: numpy datetime64
        :param b_allow_shift_back:
        :return:
        :prints: '`'/'.' when correcting small/big nonlinearity in interval
        """
        #nonlocal dt0  # to not search the local namespace first numba not implements
        for iSt, iEn, tSt, tEn, tEn_calc, tEn_err in zip(*
                                                         i_j_yi_yj_yec_yee(i_st_h[:-1], i_st_h[1:], t)):
            tEn_err_abs = abs(tEn_err)
            if tEn_err_abs < dt_big_hole * (
                    2 if b_allow_shift_back else 1):  # small correction errors (< 2*dt_big_hole)
                if __debug__:
                    print('`', end='')
                # if tEn_err_abs > dt_big_hole:  # decrease hole in half moving all interval data
                #     tSt      += (tEn_err / 2)
                #     tEn_calc += (tEn_err / 2)
                t[iSt:iEn] = np.arange(tSt, tEn_calc, dt0,
                                       np.int64)  # ValueError: could not broadcast input array from shape (1048) into shape (1044)
            else:  # big correction errors because of not uniformly distributed data inside interval
                if __debug__:
                    print('.', end='')
                if tEn_err > 0:  # some data lost inside [iSt,iEn]
                    # Reimplemented repeated2increased(t[iSt:iEn], freq, bOk)
                    # todo: remove duplicated code
                    # add increasing trend (equal to period dt0/counts) where time is not change
                    # Note: The algorithm can make some decreased (by value < dt0) elements!
                    dt_cur = np.ediff1d(t[iSt:iEn], to_begin=1)
                    bOk = (dt_cur > 0)
                    t_add = np.arange(tSt, tEn_calc, dt0, np.int64)  # uniform incremented sequence
                    # to begin incremented sequence everywhere at steps of real data w'll remove from t_add this array:
                    t_rem = np.zeros(bOk.size, dtype=np.int64)
                    t_rem[bOk] = np.ediff1d(t_add[bOk], to_begin=t_add[0])
                    t_rem = np.cumsum(t_rem)
                    t_add -= t_rem
                    # pprint(np.vstack(( bOk, t_add, np.ediff1d(t[iSt:iEn] + t_add, to_begin=0) )))
                    t[iSt:iEn] += t_add

                    # # Must not too many recursive: this hang mycomp!
                    # t_try = np.arange(tSt, tEn_calc, dt0, np.int64)
                    # t_err = t[iSt:iEn] - t_try
                    # # break current interval: retain max|last data hole and repeat
                    # iBad= np.flatnonzero(np.logical_and(t_err < dt_big_hole, dt_cur))
                    # if len(iBad):
                    #     iSplit= iSt+1+iBad[-1] #use last gap before which we can correct
                    # else:
                    #     iSplit= iSt+1+np.argmax(dt_cur)
                    # t[iSt:iSplit] = t_try[:(iSplit-iSt)]
                    # for_tim(np.hstack((iSplit, iEn)), t, b_allow_shift_back= False)
                    # pass
                else:  # increasing frequency for this interval. May be you need manualy delete repeated data?
                    # Detrmine new freq. Better estimation when
                    # First and last changed values in current interval to calc updated freq:
                    dt_cur = np.ediff1d(t[iSt:iEn], to_end=0)
                    b_inc = dt_cur > 0
                    if np.all(b_inc):
                        b_exclude = False
                    else:
                        iSt_ch = np.int64(np.searchsorted(dt_cur > 0, 0.5))
                        iEn_ch = len(dt_cur) - np.int64(np.searchsorted(np.flip(dt_cur > 0, 0), 0.5))
                        b_exclude = iEn_ch > iSt_ch  # can not use changed values if changed only one or less
                    if b_exclude:  # Excluding repeated values at edges
                        dt0new = (t[iEn_ch] - t[iSt_ch]) / (iEn_ch - iSt_ch)
                        tEn_calc = tSt + (iEn - 1 - iSt) * dt0new
                        tEn_err = tEn - tEn_calc
                    else:
                        dt0new = (tEn - tSt) / (iEn - 1 - iSt)
                        tEn_calc = tEn
                    print('increasing frequency for ', iSt, '-', iEn, ' rows. New freq = ', f'{dt64_1s / dt0new:g}Hz',
                          sep = '', end=' ')
                    t[iSt:iEn] = np.arange(tSt, tEn_calc + np.int64(dt0new / 2), dt0new, np.int64)  #
                # iBad = np.flatnonzero(abs(tim_err) > dt_big_hole)

                # tim[iSt:iEn] = tim_try
                # raise(not_implemeted)
                # plt.plot(tim_err, color='r', alpha=0.5); plt.show()
                # tim_try = np.arange(tSt, tEn, dt0, np.int64)

                # if len(tim_try)== iEn - iSt:
                #     tim[iSt:iEn]= tim_try
                # elif:

    for_tim(i_st_hole, tim)

    return True  # tim

#@njit: Invalid use of <function diff> with argument(s) of type(s): (array(float64, 1d, A)) TypingError: Internal error at <numba.typeinfer.CallConstraint object at 0x000001F83B761B08> reshape() supports contiguous array only
def is_works(s, noise=0):
    """
    Regions where data is changing
    :param s:
    :param noise: minimum adjasent data changing when working
    :return:
    """

    d = np.diff(s)
    n = np.isnan(d)
    if n.sum() > len(s)/10:
        print('too many nans - skip freezing data filter')
        return np.ones_like(s, dtype=np.bool_)

    b = np.abs(d) > noise
    for stride in [16, 32]:
        bnext = (np.abs(np.diff(s[::stride])) > noise).repeat(stride)
        b[:bnext.size] = np.logical_or(b[:bnext.size], bnext)

    return np.pad(b, (1, 0), 'edge')

#@njit: TypeError: np_unique() got an unexpected keyword argument 'return_index'
def mode(a):
    """
    stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-vector
    :param a:
    :return:
    """
    (_, i_uniq, n_uniq) = np.unique(a, return_index=True, return_counts=True)
    index = i_uniq[np.argmax(n_uniq)]
    mode = a[index]
    return index, mode

#@njit: TypeError: np_unique() got an unexpected keyword argument 'return_index'
def too_frequent_values(a, max_portion_allowed=0.5):
    """
    Find ubnormal frequent (bad) values in data and return musk where it is
    :param a: data
    :param max_portion_allowed: portion of same values safficient to delete them.
        Lower value - more filter!
    :return: mask of bad values found in a
    """

    (m_all, i_uniq, n_uniq) = np.unique(a, return_index=True, return_counts=True)
    bad_portion = 100.0 * n_uniq / a.size
    use_uniq = bad_portion > max_portion_allowed

    bbad = np.zeros_like(a, dtype=np.bool_)
    m_all = m_all[use_uniq]
    if m_all.size:
        for i, m in zip(i_uniq[use_uniq], m_all):
            bbad[a == m] = True

        print('too_frequent_values detected: ', np.around(m_all, 3),  #'[' ','.join(['{:.3g}'.format(m) for m in m_all] ']'
            '(last alternates', np.around(bad_portion[m_all.size - 1], 1), '%')  #l.info()
    else:
        if __debug__:
            if len(n_uniq):
                i = np.argmax(n_uniq)
                print('most frquent value' ,a[i_uniq[i]], 'appears', n_uniq[i], 'times - not too frequent',
                      round(bad_portion[i], 1))
            elif not len(a):
                print('- no data')
    return bbad

@njit
def too_frequent_values__old_bad(s, max_portion_allowed=0.5):
    """
    Find ubnormal frequent (bad) values in data and return musk where it is
    :param s: data
    :return: mask of bad values found in s
    """

    m = np.nanmedian(s)
    # possible bad values is fast changed or not changed (or median):
    ds = np.abs(np.ediff1d(s, to_end=0))
    bfilt = np.logical_or(np.logical_or(ds == 0, ds > 1), s == m)
    s = s[bfilt]
    bbad = np.zeros_like(s, dtype=np.bool_)
    m_all = []
    while True:
        btest = s == m
        # check that values returns to m frequently
        bad_portion = 100.0 * np.sum(np.diff(btest) != 0) / s.size
        if all(btest) or bad_portion < max_portion_allowed:
            break
        m_all.append(m)
        bbad[~bbad] |= btest
        s = s[~btest]
        m = np.nanmedian(s)
    bfilt[bfilt] = bbad
    if m_all:
        print('Detected mixed bad values:,[', ','.join([str(round(m, 3)) for m in m_all]),
              '] (last alternates ', round(bad_portion, 1), '%)' )  #l.info(
    else:
        if __debug__:
            print('median value ', m_all, 'appears', sum(btest), 'times - not too frequent (',
            round(bad_portion, 1), '% to number of different elements, or ', round(100 * sum(btest) / s.size, 1), '% to full size)')
    return bfilt


# -----------------------------------------------------------------
# Wavelet filtering
@njit
def doppler(x):
    """
    Parameters
    ----------
    x : array-like
        Domain of x is in (0,1]

    """
    if not np.all((x >= 0) & (x <= 1)):
        with objmode():
            raise ValueError("Domain of doppler is x in (0,1]")
    return np.sqrt(x * (1 - x)) * np.sin((2.1 * np.pi) / (x + .05))

@njit
def blocks(x):
    """
    Piecewise constant function with jumps at t.

    Constant scaler is not present in Donoho and Johnstone.
    """
    K = lambda x: (1 + np.sign(x)) / 2.
    t = np.array([[.1, .13, .15, .23, .25, .4, .44, .65, .76, .78, .81]]).T
    h = np.array([[4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2]]).T
    return 3.655606 * np.sum(h * K(x - t), axis=0)

@njit
def bumps(x):
    """
    A sum of bumps with locations t at the same places as jumps in blocks.
    The heights h and widths s vary and the individual bumps are of the
    form K(t) = 1/(1+|x|)**4
    """
    K = lambda x: (1. + np.abs(x)) ** -4.
    t = np.array([[.1, .13, .15, .23, .25, .4, .44, .65, .76, .78, .81]]).T
    h = np.array([[4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 2.1, 4.2]]).T
    w = np.array([[.005, .005, .006, .01, .01, .03, .01, .01, .005, .008, .005]]).T
    return np.sum(h * K((x - t) / w), axis=0)

@njit
def heavisine(x):
    """
    Sinusoid of period 1 with two jumps at t = .3 and .72
    """
    return 4 * np.sin(4 * np.pi * x) - np.sign(x - .3) - np.sign(.72 - x)


# -------------------------------------------------------------
def coef_pyramid_plot(coefs, first=0, scale='uniform', ax=None):
    """
    Shows common diagnostic plot of the wavelet coefficients

    Parameters
    ----------
    coefs : array-like
        Wavelet Coefficients. Expects an iterable in order Cdn, Cdn-1, ...,
        Cd1, Cd0.
    first : int, optional
        The first level to plot.
    scale : str {'uniform', 'level'}, optional
        Scale the coefficients using the same scale or independently by
        level.
    ax : Axes, optional
        Matplotlib Axes instance

    Returns
    -------
    Figure : Matplotlib figure instance
        Either the parent figure of `ax` or a new pyplot.Figure instance if
        `ax` is None.
    """

    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)  #, axisbg='lightgrey'
    else:
        fig = ax.figure

    n_levels = len(coefs)
    n = 2 ** (n_levels - 1)  # assumes periodic

    if scale == 'uniform':
        biggest = [np.max(np.abs(np.hstack(coefs)))] * n_levels
    else:
        # multiply by 2 so the highest bars only take up .5
        biggest = [np.max(np.abs(i)) * 2 for i in coefs]

    for i in range(first, n_levels):
        x = np.linspace(2 ** (n_levels - 2 - i), n - 2 ** (n_levels - 2 - i), 2 ** i)
        ymin = n_levels - i - 1 + first
        yheight = coefs[i] / biggest[i]
        ymax = yheight + ymin
        ax.vlines(x, ymin, ymax, linewidth=1.1)

    ax.set_xlim(0, n)
    ax.set_ylim(first - 1, n_levels)
    ax.yaxis.set_ticks(np.arange(n_levels - 1, first - 1, -1))
    ax.yaxis.set_ticklabels(np.arange(first, n_levels))
    ax.tick_params(top=False, right=False, direction='out', pad=6)
    ax.set_ylabel("Levels", fontsize=14)
    ax.grid(True, alpha=.85, color='white', axis='y', linestyle='-')
    ax.set_title('Wavelet Detail Coefficients', fontsize=16,
                 position=(.5, 1.05))
    fig.subplots_adjust(top=.89)

    return fig


# -------------------------------------------------------------
#
def waveletSmooth(y, wavelet="db4", level=1, ax=None, label=None, x=None, color='k'):
    """
    Wavelet smoothing
    :param y:
    :param wavelet: #'db8'
    :param level: 11
    :param ax:
    :param label:
    :param x: where to plot y
    :return: y, ax

    waveletSmooth(NDepth.values.flat, wavelet='db8', level=11, ax=ax, label='Depth')
    """
    if level <= 0:
        return y, ax

    # calculate the wavelet coefficients
    mode = "antireflect"
    coeff = pywt.wavedec(y, wavelet, mode=mode)  # , mode="per", level=10
    # calculate a threshold: changing this threshold also changes the behavior
    sigma = mad(coeff[-level])
    # mad = median(Wj-1, k-median(Wj-1, k)) /0.6475  # Universal freshold
    uthresh = sigma * np.sqrt(2 * np.log(len(y)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="garotte") for i in coeff[1:])  # soft

    # reconstruct the signal using the thresholded coefficients
    out = pywt.waverec(coeff, wavelet, mode=mode)
    if out.size > y.size:  # ?
        out = out[:y.size]

    if __debug__ and label:
        if x is None:
            x = np.arange(len(y))
        if ax is None:
            plt.style.use('bmh')
            f, ax = plt.subplots()
            ax.plot(x, y, color='r', alpha=0.5, label=label + ' sourse')
        ax.plot(x, out, color=color, label='{}^{}({})'.format(wavelet, level, label))
        ax.set_xlim((0, len(y)))
    else:
        ax = None
    return out, ax


# -------------------------------------------------------------------------
# Despiking
#@njit  TypeError: as_strided() got an unexpected keyword argument 'writeable'
def rolling_window(x, block):
    """
    :param x:
    :param block:
    :return:
    """
    shape = x.shape[:-1] + (x.shape[-1] - block + 1, block)
    strides = x.strides + (x.strides[-1],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)

#@njit: gaussian_filter1d() not supported
def despike(x: np.ndarray,
            offsets: Tuple[int, ...] = (2, 20),
            blocks: int = 10,
            ax=None, label='', std_smooth_sigma=None, x_plot=None
            ) -> Tuple[np.ndarray, Optional[matplotlib.axes.Axes]]:
    r"""
    Compute the statistics of the x ($\mu$ and $\sigma$) and marks (but do not exclude yet) x
    that deviates more than $n1 \times \sigma$ from the mean, Based on [https://ocefpaf.github.io/python4oceanographers/blog/2013/05/20/spikes/]
    :param x: flat numpy array (that need to filter)
    :param offsets: offsets to std. First offsets should be bigger to delete big spikes first and then filter be sensetive to more subtle errors
    :param block: filter window width
    :param ax: if not None then plots source and x averaged(blocks) on provided ax
    :param label: if not None then allow plots. If bool(label) result will be plotted with label legend
    :param std_smooth_sigma: gaussian smooth parameter, if not None std will be smoothed before multiply to offset and compare to |data - <data>|.
    :param x_plot: x data to plot y
    :return y: x with spikes replaced by NaNs
    """
    if not len(offsets):
        return x, ax
    offsets_blocks = np.broadcast(offsets, blocks)
    # instead of using NaNs because of numpy warnings on compare below
    y = np.ma.fix_invalid(x, copy=True)  # suppose the default fill value is big enough to be filtered by masked_where() below.  x.copy()
    len_x = len(x)
    std = np.empty((len_x,), np.float64)
    mean = np.empty((len_x,), np.float64)

    if __debug__:
        n_filtered = []
        if ax is not None:
            colors = ['m', 'b', 'k']
            if x_plot is None:
                x_plot = np.arange(len_x)

    for i, (offset, block) in enumerate(offsets_blocks):
        start = block // 2
        end = len_x - block + start + 1
        sl = slice(start, end)
        # recompute the mean and std without the flagged values from previous pass
        # now removing the flagged y.
        roll = np.ma.array(rolling_window(y.data, block)) if y.mask.size == 1 else (
            np.ma.array(rolling_window(y.data, block), mask=rolling_window(y.mask, block)))
        # 2nd row need because use of subok=True in .as_strided() not useful: not preserves mask (numpy surprise)
        # 1st need because y.mask == False if no masked values but rolling_window needs array

        # roll = np.ma.masked_invalid(roll, copy=False)
        roll.std(axis=1, out=std[sl])
        roll.mean(axis=1, out=mean[sl])
        std[:start] = std[start]
        mean[:start] = mean[start]
        std[end:] = std[end - 1]
        mean[end:] = mean[end - 1]
        assert std[sl].shape[0] == roll.shape[0]
        if std_smooth_sigma:
            std = gaussian_filter1d(std, std_smooth_sigma)

        y = np.ma.masked_where(np.abs(y - mean) > offset * std, y, copy=False)

        if __debug__:
            n_filtered.append(y.mask.sum())
            if ax is not None:
                ax.plot(x_plot, mean, color=colors[i % 3], alpha=0.3,
                        label='{}_mean({})'.format(label if label is not None else '', block))
    y = np.ma.filled(y, fill_value=np.NaN)
    if __debug__:
        print('despike(offsets=', offsets, ', blocks=', blocks, ') deletes', n_filtered, ' points')
        if ax is not None:
            ax.plot(x_plot, y, color='g', label=f'despike{blocks}{offsets}({label})')
            ax.set_xlim((0, len(y)))

    return y, ax

#@njit: Use of unsupported NumPy function 'numpy.einsum' or unsupported use of the function
def closest_node(node, nodes):
    """
    see also inearestsorted()
    :param node: point (2D)
    :param nodes: 2D array
    :return:
    Based on http://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points/28210#28210
    """
    # nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->j', deltas, deltas)
    return np.argmin(dist_2)

#@njit "Invalid use of function searchsorted with argument(s) of type(s): (array(datetime64[ns], 1d, C), array(datetime64[ns], 1d, C))"
def inearestsorted(array, values):
    """
    Find nearest values in sorted numpy array
    :param array:  numpy array where to search, sorted
    :param values: numpy array to which values need to find nearest
    :return: found indexes of first array size of values
    """
    if not array.size:
        return []
    idx = np.searchsorted(array, values)  # side defaults to "left"
    idx_prev = np.where(idx > 0, idx - 1, 0)
    idx = np.where(idx < array.size, idx, array.size - 1)
    return np.where(np.abs(values - array[idx_prev]) <
                    np.abs(values - array[idx]), idx_prev, idx)


# def inearest_innotsored(array, value):
#     # stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
#     idx_sorted = np.argsort(array)
#     sorted_array = np.array(array[idx_sorted])
#     idx = np.searchsorted(sorted_array, value, side="left")
#     if idx >= len(array):
#         idx_nearest = idx_sorted[len(array)-1]
#     elif idx == 0:
#         idx_nearest = idx_sorted[0]
#     else:
#         if abs(value - sorted_array[idx-1]) < abs(value - sorted_array[idx]):
#             idx_nearest = idx_sorted[idx-1]
#         else:
#             idx_nearest = idx_sorted[idx]
#     return idx_nearest

#@njit: Invalid use of Function(<function searchsorted at 0x0000021673822E58>) with argument(s) of type(s): (array(datetime64[ns], 1d, C), array(datetime64[ns], 1d, C))
def inearestsorted_around(array, values):
    """
    Find nearest values before and after of each values in sorted numpy array
    Returned values useful as indexes for linear interpolation of data associated with values
    :param array:  numpy array where to search
    :param values: numpy array to which values need to find nearest
    :return: found indexes of length <= len(values)
    """
    if not array.size:
        return []
    idx = np.searchsorted(array, values)  # side defaults to "left"
    idx_prev = np.where(idx > 0, idx - 1, 0)
    idx = np.where(idx < array.size, idx, array.size - 1)
    idx2 = np.sort(np.hstack((idx_prev, idx)))
    idx2 = idx2[np.ediff1d(idx2, to_end=1) > 0]  # remove repeated
    return idx2

@njit
def search_sorted_closest(sorted_array, value):
    """
    simpler version of inearestsorted() for scalar value
    :param sorted_array:
    :param value:
    :return: index of cosest element in sorted_array

    Notes: not tested
    """
    inav_close = np.search_sorted(sorted_array, value)
    # search_sorted to search closest
    if inav_close < sorted_array.size:
        inav_close += np.argmin(sorted_array[inav_close + np.int32([0, 1])])
    else:
        inav_close -= 1
    return inav_close


def check_time_diff(t_queried: Union[pd.Series, np.ndarray], t_found: Union[pd.Series, np.ndarray],
                    dt_warn: Union[pd.Timedelta, np.timedelta64],
                    mesage: str = 'Bad nav. data coverage: difference to nearest point in time [min]:',
                    return_diffs: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Check time difference between found and requested time points and prints info if big difference is found
    :param t_queried: pandas TimeSeries or numpy array of 'datetime64[ns]'
    :param t_found:   pandas TimeSeries or numpy array of 'datetime64[ns]'
    :param dt_warn: pd.Timedelta - prints info about bigger differences found only
    :param return_diffs: if True also returns time differences (t_found - t_queried_values, 'timedelta64[ns]')
    :return: mask where time difference is bigger than ``dt_warn`` and time differences if return_diffs=True
    """
    try:
        if not np.issubdtype(t_queried.dtype, np.dtype('datetime64[ns]')):  # isinstance(, ) pd.Ti
            t_queried_values = t_queried.values
        else:
            t_queried_values = t_queried
    except TypeError:  # not numpy 'datetime64[ns]'
        t_queried_values = t_queried.values

    dT = np.array(t_found - t_queried_values, 'timedelta64[ns]')
    bbad = abs(dT) > np.timedelta64(dt_warn)
    if (mesage is not None) and np.any(bbad):
        if mesage:
            mesage = '\n'.join(['mesage'] + ['{}. {}:\t{}{:.1f}'.format(
                i, tdat, m, dt / 60) for i, tdat, m, dt in zip(
                np.flatnonzero(bbad), t_queried[bbad], np.where(dT[bbad].astype(np.int64) < 0, '-', '+'),
                np.abs(dT[bbad]) / np.timedelta64(1, 's'))])
        l.warning(mesage)
    return (bbad, dT) if return_diffs else bbad


# ##############################################################################
if __name__ == '__main__':
    from scipy import stats

    """
    Generate the y and get the coefficients using the multilevel discrete wavelet transform.
    """
    #
    # db8 = pywt.Wavelet('db8')
    # scaling, wavelet, x = db8.wavefun()

    np.random.seed(12345)
    blck = blocks(np.linspace(0, 1, 2 ** 11))
    nblck = blck + stats.norm().rvs(2 ** 11)

    true_coefs = pywt.wavedec(blck, 'db8', level=11, mode='per')
    noisy_coefs = pywt.wavedec(nblck, 'db8', level=11, mode='per')

    # Plot the true coefficients and the noisy ones.
    fig, axes = plt.subplots(2, 1, figsize=(9, 14), sharex=True)

    _ = coef_pyramid_plot(true_coefs[1:], ax=axes[0])  # omit smoothing coefs
    axes[0].set_title("True Wavelet Detail Coefficients")

    _ = coef_pyramid_plot(noisy_coefs[1:], ax=axes[1])
    axes[1].set_title("Noisy Wavelet Detail Coefficients")

    fig.tight_layout()

    """
    Apply soft thresholding using the universal threshold
    """
    # robust estimator of the standard deviation of the finest level detail coefficients:
    sigma = mad(noisy_coefs[-1])
    uthresh = sigma * np.sqrt(2 * np.log(len(nblck)))

    denoised = noisy_coefs[:]

    denoised[1:] = (pywt.thresholding.soft(i, value=uthresh) for i in denoised[1:])

    """
    Recover the signal by applying the inverse discrete wavelet transform to the thresholded coefficients
    """
    signal = pywt.waverec(denoised, 'db8', mode='per')

    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True,
                             figsize=(10, 8))
    ax1, ax2 = axes

    ax1.plot(signal)
    ax1.set_xlim(0, 2 ** 10)
    ax1.set_title("Recovered Signal")
    ax1.margins(.1)

    ax2.plot(nblck)
    ax2.set_title("Noisy Signal")

    for ax in fig.axes:
        ax.tick_params(labelbottom=False, top=False, bottom=False, left=False,
                       right=False)

    fig.tight_layout()

@njit()
def find_sampling_frequency(tim: np.ndarray,
                            precision: float = 0,
                            b_show: bool = True) -> Tuple[np.float_, int, int, np.ndarray]:
    """
    Function tries to find true frequency inside packets.
    Finds highest frequency of irregular array _tim_ with specified precision.
    Checks time for decreasing and nonicreasing - prints messages
    :param tim: time, datetime64 or int64
    :param precision: If tim in int64 nanoseconds then set precision in [log10(s)]
    :param b_show:
    :return: (freq, n_same, n_dec, b_ok):
        freq: frequency, Hz
        n_same: number of nonincreased elements found
        n_dec: number of decreased elements found
        i_inc: int, indexes of increased tim elements

       Prints n_dec ' time inversions detected!'
       and n_same '% nonincreased'
    """

    # diff(tim) should be approcsimately constant but can have some big spikes and many zeros
    # We have tipically the folowing time difference dt between counts:
    # ^ dt
    # |    |             |             |
    # | 0 _|____|____|___|____|____|___|_
    #     ............................... counts
    #    xx          xxxx          xxxx - what we should exclude to average all other dt*
    # where zeros because bad data resolutions and big values because of data recorded in packets
    #
    # *If time generated by device start of packet is measured exactly
    n_dec = np.int64(0)  # type is specified here for numba only
    n_same = np.int64(0)
    if tim.size < 2:
        with objmode():
            l.warning('can not find freq')
        return np.NaN, n_same, n_dec, np.zeros(1, np.int64)  # zeros() is for numba only

    dt = np.ediff1d(tim.view(np.int64), to_end=1)
    # all increased positions
    b_ok = dt > 0
    i_inc_out = np.flatnonzero(b_ok)  # where time is increased (allowed by time resolution)
    bBad = dt < 0
    bAnyDecreased = bBad.any()
    if bAnyDecreased:
        n_dec = bBad.sum()
        print(n_dec, 'time inversions detected!')
        # ignore all this:
        # - joined different data in backward order
        # - data with different frequencies
        bBad = b1spike(tim)
        dt = np.ediff1d(tim[~bBad].view(np.int64), to_end=1)
        bBad = dt < 0
        bAnyDecreased = bBad.any()
        if bAnyDecreased:  # b1spike can not remove time decreasing
            dt = dt[~bBad]
            n_dec = bBad.sum()
            print(n_dec, 'time inversions that are not single spikes!')
            
        b_ok = dt > 0  # filtered increased positions
        i_inc = np.flatnonzero(b_ok)  
    else:
        i_inc = i_inc_out
    
    
    # Find freq

    n_inc = b_ok.sum()
    if n_inc < 2:
        with objmode():
            l.warning("Can't find freq if Time have less than 2 increased values!")
        return np.NaN, n_same, n_dec, np.zeros(1, np.int64)

    n_nondec = dt.size - n_inc
    if n_nondec > 0:
        n_nondec_r = n_nondec / np.float64(dt.size)
        print(round(100 * n_nondec_r, 1), '% nonincreased...')

    # number of elements between increased elements
    di_inc = np.ediff1d(i_inc, 0)  # areas lengths, last is unknown is set to 0 but value will not be used
    # not packets starts mask
    dt_f = dt[b_ok]
    b_f = dt_f < (dt_f.mean() + dt64_1s * 10 ** precision)
    # exclude diffs corresponding to packets and also diffs to their starts & from ends because their starts and duration may be not in sync with time diffs of small resolution we dealing:
    b_f[1:] &= b_f[:-1]
    b_f[:-1] &= b_f[1:]
    b_f[-1] = False  # excluding last area which length is unknown (could be cut)

    freqs = np.float64(dt64_1s) * (di_inc / dt_f)[b_f]  # Hz, frequencies between
    medFr = np.median(freqs)                              # get only most frequently used tim period
    stdFr = freqs.std()
    bGood = np.logical_and(medFr - stdFr * 3 < freqs, freqs < medFr + stdFr * 3)  # filtConstraints(freqs, minVal= medFr/3, maxVal= medFr*3)

    # mean(median smothed frequency of data) with frequency range [med_dt/3, med_dt*3]:
    if bGood.sum() > 1:
        freqs = freqs[bGood]
    freq = freqs.mean()
    freq = round(freq, int(max(precision, int(np.log10(freq)))))  # never round to 0
    if b_show:
        print('freq =', freq, 'Hz')
    return freq, n_nondec - n_dec, n_dec, i_inc_out


"""
    dt64_1s = np.int64(1e9)
    n_decrease = 0
    n_same_r = 0
    if Time.size < 2:
        freq = np.NaN
        b_ok = np.ones(1, dtype=np.bool_)
    else:
        dt = np.pad(np.diff(Time), (0, 1), 'mean')
        b_ok = dt > 0  # all increased positions
        bBad = np.zeros_like(dt, dtype=np.bool)
        for i in range(10):
            bBad[~bBad] = dt < 0
            bAnyDecreased = np.any(bBad)
            if bAnyDecreased:
                n_decrease = np.sum(bBad)
                print(str(n_decrease) + ' time inversions detected!')
                # ignore all this:
                # - may be splitted different data in backward order
                # - may be this data with different frequencies
                if i%2:
                    dt = np.ediff1d(Time[~bBad], to_end=True)
                else:
                    dt = np.ediff1d(Time[~bBad], to_begin=True)
            else:
                break
        b_ok[bBad] = False
        n_same = np.size(dt) - np.sum(b_ok)
        bAnyNonIncreased = n_same > 0
        if bAnyNonIncreased:
            n_same_r = np.float64(n_same) / np.float64(dt.size)
            print(str(round(100 * n_same_r, 1)) + '% nonincreased elements considered')
        i_inc = np.flatnonzero(~bBad[b_ok&~bBad])  # where time increased (allowed by time resolution)
        dt_f = dt[dt>0]
        b_f = dt_f < (dt_f.mean() + dt64_1s * 10 ** precision)  # not packet starts in i_inc array
        # number of elements between increased elements excluding packets starts:
        di_inc = np.diff(np.hstack((-1, i_inc)))[b_f]
        freqs = np.float64(dt64_1s) * di_inc / dt[i_inc[b_f]]  # Hz, frequencies between
        medFr = np.median(freqs)  # get only most frequently used Time period
        stdFr = np.std(freqs)
        bGood = np.logical_and(medFr - stdFr * 3 < freqs,
                               freqs < medFr + stdFr * 3)  # filtConstraints(freqs, minVal= medFr/3, maxVal= medFr*3)

        # mean(median smothed frequency of data) with frequency range [med_dt/3, med_dt*3]:
        if np.sum(bGood) > 1:
            freqs = freqs[bGood]
        freq = np.mean(freqs)
        freq = np.round(freq, int(max(precision, np.fix(np.log10(freq)))))  # never round to 0
    if b_show:
        print('freq = ' + str(freq) + 'Hz')
    return freq, n_same_r, n_decrease, b_ok
"""
