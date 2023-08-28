import logging
from typing import Any, Mapping

import numpy as np

# my

l = logging.getLogger(__name__)


def dist_clc(track, a1DLat, a1DLon, a1DTime, cfg: Mapping[str, Any]):  # , useLineDist
"""
Calculate Distanse(a1DLat, a1DLon) = trackDist (same size as a1DLat, a1DLon, a1DTime). If nargout>1 selects good points for calc. distance  add struct "Points" to track. "Points" has fields: "Dist","Time","Lat","Lon"
:param track:
:param a1DLat:
:param a1DLon:
:param a1DTime:
:return: [trackDist, track, a1DLat, a1DLon]
"""

if 'useLineDist' in cfg['track']:
    track['useLineDist'] = 0.05
#%% check first/ track points
b_use = np.isfinite(a1DLat) & np.isfinite(a1DLon) & np.isfinite(a1DTime)
iEdge = np.flatnonzero(b_use)([0, -1])
if iEdge.size:
    trackDist = np.NaN
if not a1DTime.size:
    l.warning('DistClc:NoData - empty track')
else:
    l.warning('DistClc:NoData - no Lat,Lon for track starts at {}'.format(a1DTime[0]))

track['Points']['Time'] = a1DTime(b_use)
track['Points']['Dist'] = np.NaN(b_use.size)
track['Points']['Lat'] = np.NaN(b_use.size)
track['Points']['Lon'] = np.NaN(b_use.size)
return

bnan = ~np.isfinite(a1DLat)
k = np.sum(bnan)
if k > 0:
    print('#g np.NaNs found in Lat!!! Check navInterp\n', k)
    a1DLat = rep2mean(a1DLat, bnan, a1DTime)

bnan = ~np.isfinite(a1DLon)
k = np.sum(bnan)
if k > 0:
    print('#g np.NaNs found in Lon!!! Check navInterp\n', k)
    a1DLon = rep2mean(a1DLon, bnan, a1DTime)

b_use = ~(~np.isfinite(a1DLat) | ~np.isfinite(a1DLon) | ~np.isfinite(a1DTime))
iEdge = np.flatnonzero(b_use)([0, -1])

if (not track.size) or (not 'LatSt' in track) or (not track['LatSt'].size) or ~np.isfinite(track['LatSt']):
    track['Ens'] = 1
    if not 'bInvert' in track: track['bInvert'] = False;
    if track['bInvert']:
        k = iEdge()
        p = iEdge[0]
    else:
        k = iEdge[0]
        p = iEdge()

    track['LatSt'] = a1DLat(k)
    track['LatEn'] = a1DLat[p]
    track['LonSt'] = a1DLon(k)
    track['LonEn'] = a1DLon[p]

if np.sum(b_use) < 2:
    trackDist = 0
    # if nargout>1:
    track['Points']['Lat'] = a1DLat[iEdge]
    track['Points']['Lon'] = a1DLon[iEdge]

elif np.isfinite(track['useLineDist']):
    ## Curve distance
    # np.flatnonzero b_use where dDist>track['useLineDist']
    trackDist = legs(a1DLat[b_use], a1DLon[b_use])[1]  # nm
    b_use[iEdge[0]] = False
    dDist = np.full_like(a1DLat, np.NaN)
    dDist[b_use] = trackDist
    dDist[~np.isfinite(dDist)] = 0
    trackDist = 0
    useLineDist = km2nm(track['useLineDist'])
    for p in np.flatnonzero(b_use):
        trackDist = trackDist + dDist[p]
        b_use[p] = (trackDist >= useLineDist)
        if b_use[p]:
            trackDist = 0

    b_use[iEdge] = True
    trackDist = legs(a1DLat[b_use], a1DLon[b_use])[1]
    b_use[iEdge[0]] = False
    dDist = np.full_like(a1DLat, np.NaN)
    dDist[b_use] = trackDist
    dDist[iEdge[0]] = 0
    b_use[iEdge[0]] = True
    trackDist = np.full_like(a1DLat, np.NaN)
    trackDist[b_use] = np.cumsum(dDist[b_use])
    if track['bInvert']:
        trackDist = trackDist(iEdge(2)) - trackDist

    if 'bMeanOnIntervals' in track and track['bMeanOnIntervals'] and np.sum(b_use) > 2:
        # better accuracy points, especially for speed calc:
        kLast = 1;
        p = 1;
        b_use[0] = False
        temp = np.flatnonzero(b_use) + 1
        track['Points']['Time'] = np.NaN(temp.size)
        track['Points']['Lat'] = np.NaN(temp.size)
        track['Points']['Lon'] = track['Points']['Lat']
        for k in temp.T:
            ind = slice(kLast, (k - 1))
            track['Points']['Time'][p] = mean(a1DTime(ind))
            track['Points']['Lat'][p] = np.nanmean(a1DLat(ind))
            track['Points']['Lon'][p] = np.nanmean(a1DLon(ind))
            kLast = k
            p += 1

        # lat1= track['Points']['Lat'][0]+diff(track['Points']['Lat']([1 2]))*( # (a1DTime[0]-track['Points']['Time'][0])/diff(track['Points']['Time']([1 2]))
        # lon1= track['Points']['Lon'][0]+diff(track['Points']['Lon']([1 2]))*( # (a1DTime[0]-track['Points']['Time'][0])/diff(track['Points']['Time']([1 2]));
        track['Points']['Dist'] = legs(track['Points']['Lat'], track['Points']['Lon'])[1]
        track['Points']['Dist'] = [0, np.cumsum(track['Points']['Dist'])]
        trackDist = np.interp(track['Points']['Time'], track['Points']['Dist'], a1DTime)

    if track['bInvert']:
        trackDist = trackDist(iEdge(2)) - trackDist
    else:
        track['bMeanOnIntervals'] = False
        trackDist = rep2mean(trackDist, ~b_use, a1DTime)
        trackDist = nm2km(trackDist)

        track['Points']['Lat'] = a1DLat[b_use]
        track['Points']['Lon'] = a1DLon[b_use]

else:  # line distance
    track['Course'] = legs([track['LatSt'], track['LatEn']], [track['LonSt'], track['LonEn']], 'gc')[0]
    az = track['Course'] * np.pi / 180
    lat1 = track['LatSt'] * np.pi / 180
    lon1 = track['LonSt'] * np.pi / 180
    lon1 = lon1 + np.atan2(np.cos(az), np.sin(az) * np.sin(lat1))
    lon1 = np.pi * ((abs(lon1) / np.pi) - 2 * np.ceil(((abs(lon1) / np.pi) - 1) / 2)) * sign(lon1)
    lat1 = np.asin(-np.cos(lat1) * np.sin(az))
    lats = a1DLat[b_use] * np.pi / 180
    lons = a1DLon[b_use] * np.pi / 180
    lons = lons + np.atan2(-np.sin(az), np.cos(az) * np.sin(lats))
    lons = (abs(lons) - 2 * np.pi * np.ceil(((abs(lons) / np.pi) - 1) / 2)) * np.sign(lons)
    lats = np.asin(-np.cos(lats) * np.cos(az))
    lats, lons = scxsc(repmat(lat1, lats.size), repmat(lon1, lats.size), repmat(np.pi / 2, lats.size), lats, lons,
                       repmat(np.pi / 2, lats.size), 'radians')
    b_ind = lats > 0;
    lats = lats(b_ind) * 180 / np.pi
    b_ind = lons > 0;
    lons = lons(b_ind) * 180 / np.pi

    (temp, dDist) = legs([track['LatSt'], lats], [track['LonSt'], lons], 'gc')

    dDist = dDist * (2 * (abs(temp - track['Course']) < 90) - 1)  # (* sign)
    dDist = np.cumsum(dDist)
    trackDist = np.full_like(a1DLat, np.NaN)
    trackDist[b_use] = dDist
    track['Points']['Lat'] = lats
    track['Points']['Lon'] = lons

    trackDist = nm2km(trackDist)  # convert & cat if was longer in cycle before
# DistLine
if not track['bMeanOnIntervals']:
    track['Points']['Dist'] = trackDist[b_use]
    track['Points']['Time'] = a1DTime[b_use]
    if (iEdge[0] != 1) or (iEdge(2) != numel(a1DLon)):
        track['EnsEnd'] = track['Ens'] + iEdge(2) - 1
        track['Ens'] = track['Ens'] + iEdge[0] - 1
