import pyproj
def to_utm(latitude, longitude):
    utm_proj = pyproj.Proj(proj='utm', zone=34, ellps='WGS84', south=False)
    utm_easting, utm_northing = utm_proj(longitude, latitude)
    return utm_easting, utm_northing

def utm_to_degrees(easting, northing, zone_number=34):
    utm_proj = pyproj.Proj(proj='utm', zone=zone_number, ellps='WGS84', south=False)
    lon, lat = utm_proj(easting, northing, inverse=True)
    return lat, lon

if False:

    utm = [to_utm(lat, lon) for lat, lon in (
    (55.44, 10.97),
    (55.21, 11.08),
    (54.76, 10.81),
    (54.58, 11.30),
    (54.40, 11.66),
    (54.45, 12.09),
    (54.70, 12.21),
    (54.86, 12.46),
    (54.94, 13.17),
    (54.96, 13.87),
    (55.24, 14.53),
    (55.44, 14.63),
    (55.54, 14.96),
    (55.47, 15.59),
    (55.37, 16.22),
    (55.26, 16.62),
    (55.21, 17.33),
    (55.31, 17.83),
    (55.34, 18.74),
    (55.90, 19.13)
    )]


    degs  = [utm_to_degrees(e, 6017061, zone_number=34) for e, n in utm]
    degs

degs  = [utm_to_degrees(e, n) for e, n in (
         (54.6966077981, 20.4351912148),
         (54.6960472794, 20.4348846125),
         (54.6963667052, 20.4360684258)
         )]


{**{f'_{k}': ''.join([
    f" {(b - bd):.0f}{I['m']}" ,
    '^{\color{blue}{' +
    ','.join([
        fr'{si}{ki[int(ki.startswith("i")):]}' for ki, si in zip(
            k.split('_'), ([''] if k.startswith('p') else s.split(','))
        )
    ]) +
    '}'
]) for k_, (p, b, bd, s, *kw) in DISPdevices_info.items()  for k in (
    [k_, k_.replace('ip', 'p')] if k_.startswith('ip') else [k_])
},
'_Wind': v.c1(I['wind'])
}



{**{f'_{k}': ''.join([, f" {(b - bd):.0f}{I['m']}" , '^{\color{blue}{' + ','.join([fr'{si}{ki[int(ki.startswith("i")):]}' for ki, si in zip(k.split('_'), ([''] if k.startswith('p') else s.split(',')))]) + '}}']) for k_, (p, b, bd, s, *kw) in DISPdevices_info.items()  for k in ([k_, k_.replace('ip', 'p')] if k_.startswith('ip') else [k_])}, '_Wind': v.c1(I['wind'])}