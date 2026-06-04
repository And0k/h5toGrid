import netCDF4

nc = netCDF4.Dataset(
    r"B:\Cruises\BalticSea\201202_BalticSpit\inclinometers\201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6\201202P1-5,I1-2.proc_noAvg_proc_psd.nc",
    "r"
)
print("Groups:", list(nc.groups.keys()))
if "psd" in nc.groups:
    print("psd group variables:", list(nc.groups["psd"].variables.keys()))
    print("psd group dimensions:", list(nc.groups["psd"].dimensions.keys()))
    for var in nc.groups["psd"].variables:
        v = nc.groups["psd"].variables[var]
        print(f"  {var}: shape={v.shape}, dims={v.dimensions}")
nc.close()
