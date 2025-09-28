from math import ceil, inf

# The minimum recording interval, IMin, is calculated using parameters:
iPings = 100  # 100; 150; 200; 250; 300; 400; 500; 600; 800; 1000; 1200; 1500; 2000; 2500; 3000
PulseLength = 1.0  # 1.0 .. 5.0 in steps of 0.1
# Total extent of all the profile columns in meters.
D = 17  # ApproximateDistanceFromSeabedToSurface must be between 0m and 2000m. (is this relevant to define D?)
CellSize = PulseLength  # or what?
# Total number of cells in the system:
TNC = ceil(D / CellSize)  # or what?

IMinPossible = [30, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600, 5400, 7200, 9000, 10800, 14400, 21600, 28800]
print('Minimum recording interval in seconds: ')
for PowerLevel in ['High', 'Low']:
    Tc1 = PulseLength * 72000 / 750 - 50 if PowerLevel == 'High' else \
        PulseLength * 45000 / 750 - 50 if PowerLevel == 'Low' else \
            25  # if PowerLevel==Off
    Tc = max(25, Tc1)
    Tp = PulseLength * 1000 / 750
    Ts = D * 1.15 * 1000 / 750
    Tf = 1.2 * D + 4
    Tdsp = 0.86 * CellSize * TNC + 0.05 * TNC + 6
    Tsa = 7.0 * TNC + 90
    mp = 250 if PowerLevel != 'High' else \
        max(PulseLength * 100 / 5, 250) if PowerLevel == 'High' and PulseLength > 0.5 else \
            inf  # (the worst) or what?

    p0 = (Tc + Tp + Ts + Tf + Tdsp + Tsa) * 1.05
    p = max(p0, mp) / 1000
    P = ceil(p) if p > 1 else ceil(p * 100) / 100

    # Minimum recording interval in seconds:
    IMin = ceil(P * iPings)

    for m in IMinPossible:
        if m >= IMin:
            IMin = m
            break
    else:
        IMin = IMinPossible[-1]
    print('{IMin} for PowerLevel = {PowerLevel}'.format(**{'IMin': IMin, 'PowerLevel': PowerLevel}))

"""
Еще характеристик RDCP:
The default pulse length is set as the shortest cell used in all the columns. If UseDefaultPingSetup is set to 0 (false), the pulse length can be changed to values, m:
PulseLength = 1.0 .. 5.0 in steps of 0.1

If the instrument is deployed up side down, surface referenced columns are not allowed, neither is the surface cell. 
At least one column or a surface cell must be present in the configuration. If the configuration is not using any profile columns it should rather disable the profile sensor by using the set_property_profile_active(inactive) command. 
The cell sizes ranges from 1.0m to 10.0m in steps of 0.1m. If setting up a surface referred column, the minimum distance from the sea surface to the nearest cell is:
MinStartFirstCell =
= PulseLength/2 + MinBlankingDistTransducerHP	| if Highpower
= PulseLength/2 + MinBlankingDistTransducerLP	| if Lowpower
If the use surface cell is set to 0 (false), the surface cell size must be 0.0. 

Selectable instrument depths, in meters: 5; 6; 7; 8; 9; 10; 12; 15; 20; 25; 30; 35; 40; 45; 50; 55; 60; 65; 70; 75; 80; 85; 90; 95; 100; 150; 200; 300; 400; 500; 600; 700; 800; 900; 1000; 1500; 2000 

Selectable numbers of pings in one record are:
iPings = 100; 150; 200; 250; 300; 400; 500; 600; 800; 1000; 1200; 1500; 2000; 2500; 3000
If the pressure is updated at the recording instance only, the custom update interval must be set to 0 (false).

Selectable custom set pressure update intervals are (in seconds): 40; 50; 60; 70; 80; 90; 100; 110; 120; 130; 140; 150; 160; 170; 180; 190; 200; 210; 220; 230; 240; 250; 260; 270; 280; 290; 300

The minimum recording interval allowed in seconds is given by IMin below, however, the selectable recording intervals in seconds are:
IMin = 30; 60; 120; 180; 300; 600; 900; 1200; 1800; 3600; 5400; 7200; 9000; 10800; 14400; 21600; 28800 if above or equal to the minimum recording interval.

The minimum recording interval, IMin, is calculated as follows in the 4 steps below (IMin is given in step 4):
Using parameters:
D: Total extent of all the profile columns in meters. (ApproximateDistanceFromSeabedToSurface must be between 0m and 2000m. (is this relevant to define D?)
TNC: Total number of cells in the system.

1.
p0 = (Tc + Tp + Ts + Tf + Tdsp + Tsa)*1.05
where
Tc1 =
= PulseLength*72000/750 - 50	| if PowerLevel High
= PulseLength*45000/750 - 50	| if PowerLevel Low
= 25							| if PowerLevel Off

Tc = max(25, Tc1)

Tp = PulseLength * 1000/750

Ts = D * 1.15 * 1000/750

Tf = 1.2 * D + 4

Tdsp = 0.86 * CellSize * TNC + 0.05 * TNC + 6 

Tsa = 7.0 * TNC + 90

mp = 250 							| if PowerLevel != High
mp = max(PulseLength*100/5, 250)	| if PowerLevel = High and PulseLength > 0.5

2.
p = max(p0, mp)/1000

3.
P 	= ceil(p)			| if p >  1
P 	= ceil(p*100)/100	| if p <= 1

4. Minimum recording interval in seconds:
IMin = ceil((P)*iPings)



Selectable recording periods are, in hours:
1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12; 13; 14; 15; 16; 17; 18; 19; 20; 21; 22; 23. The recording period must, however, be longer or equal to the recording interval. 

Selectable number of samples in calculating the wave parameter is: 128; 256; 512; 1024; 2048; 4096
"""
