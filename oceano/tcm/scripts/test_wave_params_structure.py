#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify wave parameters NetCDF structure.

This script checks that wave parameters are saved with separate datasets
for each table, as expected.
"""
import netCDF4
from pathlib import Path


def check_wave_params_structure(nc_path: Path) -> None:
    """
    Check the structure of wave parameters NetCDF file.

    Args:
        nc_path: Path to wave parameters NetCDF file
    """
    if not nc_path.exists():
        print(f"Error: NetCDF file not found: {nc_path}")
        return

    print(f"\n{'='*70}")
    print(f"Checking wave parameters structure: {nc_path}")
    print(f"{'='*70}\n")

    with netCDF4.Dataset(nc_path, mode="r") as nc:
        print(f"Root groups: {list(nc.groups.keys())}")
        print(f"\nRoot attributes: {dict(nc.attrs)}\n")

        # Check each table group
        for group_name in nc.groups:
            group = nc.groups[group_name]
            print(f"\n{'-'*70}")
            print(f"Group: {group_name}")
            print(f"{'-'*70}")

            print(f"Variables: {list(group.variables.keys())}")
            print(f"Dimensions: {dict(group.dimensions)}")
            print(f"Attributes: {dict(group.attrs)}")

            # Check for time_domain subgroup
            if "time_domain" in group.groups:
                time_domain_group = group.groups["time_domain"]
                print(f"\n  Subgroup: time_domain")
                print(f"  Variables: {list(time_domain_group.variables.keys())}")
                print(f"  Dimensions: {dict(time_domain_group.dimensions)}")
                print(f"  Attributes: {dict(time_domain_group.attrs)}")

            # Print variable details
            for var_name in group.variables:
                var = group.variables[var_name]
                print(f"\n  Variable: {var_name}")
                print(f"    Shape: {var.shape}")
                print(f"    Dimensions: {var.dimensions}")
                print(f"    Attributes: {dict(var.attrs)}")

    print(f"\n{'='*70}")
    print("Structure check completed")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Test with a sample wave params file
    # Adjust path as needed
    nc_path = Path("B:/Cruises/BalticSea/201202_BalticSpit/inclinometer/"
                   "201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6/"
                   "201202P1-5,I1-2.proc_noAvg_wave_params.nc")

    check_wave_params_structure(nc_path)
