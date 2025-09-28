from sympy import *

# Checking that Veusz new dir:
# (360 + degrees(arctan2(Hxyz[2,:] - Gxyz[2,:]*sum(Hxyz*Gxyz, 0), Gxyz[1,:]*Hxyz[0,:] - Gxyz[0,:]*Hxyz[1,:])))%360 - 180
# equal to
# fHeading(Hxyz,sPitch,sRoll) - arctan2(tan(sRoll), tan(sPitch))

ax, ay, az, mx, my, mz = symbols('ax, ay, az, mx, my, mz')
p, r = symbols('p, r')
fpitch = -atan2(ax, sqrt(ay ** 2 + az ** 2))
froll = atan2(ay, az)
fheading = atan2(mz * sin(r) - my * cos(r), mx * cos(p) + (my * sin(r) + mz * cos(r)) * sin(p))
simplify(fheading.subs(r, froll).subs(p, fpitch) - atan2(tan(froll), tan(fpitch)))
# -atan2(ay/az, -ax/sqrt(ay**2 + az**2)) + atan2((ay*mz - az*my)/sqrt(ay**2 + az**2), -(ax*(ay*my + az*mz) - mx*(ay**2 + az**2))/(sqrt(ay**2 + az**2)*sqrt(ax**2 + ay**2 + az**2)))
# factor(fheading.subs(r, froll).subs(p, fpitch) - atan2(tan(froll), tan(fpitch)))
trigsimp(
    pi / 2 - atan2(ay / az, -ax / sqrt(ay ** 2 + az ** 2)) +
    atan2((ay * mz - az * my) / sqrt(ay ** 2 + az ** 2),
          -(ax * (ay * my + az * mz) - mx * (ay ** 2 + az ** 2)) / (
                  sqrt(ay ** 2 + az ** 2) * sqrt(ax ** 2 + ay ** 2 + az ** 2))), method='groebner')
