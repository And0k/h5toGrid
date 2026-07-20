#!/usr/bin/env python3
"""
Export inclinometer coefficients from HDF5 file to individual YAML files.

Reads coefficients from an HDF5 store (pytables via pandas) and writes each
probe's coefficients to a separate YAML file compatible with
``tcm.incl_calc.coefs.get_coefs_from_cfg()``.

Usage
-----
    python scripts/export_coefs_to_yaml.py [h5_path] [output_dir]

Arguments
---------
    h5_path: Path to HDF5 file with coefficients.
    output_dir: Directory for YAML files.

Each output file is named ``{tbl}.yaml`` where tbl is the probe table name (e.g. ``incl00.yaml``, ``incl_p01.yaml``).
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tcm.incl_calc.coefs import load_coefs
import numpy as np
import ruamel.yaml


def list_probe_tables(h5_path: Path):
    """Return list of probe table names (groups with a 'coef' child) in the HDF5 file.

    Coefficients are plain HDF5 arrays (not pandas tables), so pd.HDFStore.keys()
    won't see them. Uses get_node() to walk the HDF5 group hierarchy, consistent
    with how load_coefs() accesses ``{tbl}/coef``.
    """
    import pandas as pd
    import tables  # pytables — same backend as pd.HDFStore

    tables_list = []
    with pd.HDFStore(h5_path, mode="r") as s:
        root = s.get_node("/")
        for group_name in root._v_children:
            try:
                coef_node = s.get_node(f"/{group_name}/coef")
            except tables.exceptions.NoSuchNodeError:
                continue
            if coef_node is not None:
                tables_list.append(group_name)
    return sorted(tables_list)


def export_coefs(h5_path: Path, output_dir: Path):
    """Export all probe coefficients from HDF5 to YAML files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = list_probe_tables(h5_path)
    if not tables:
        print(f"No coefficient tables found in {h5_path}")
        return

    ry = ruamel.yaml.YAML(typ="safe", pure=True)
    ry.default_flow_style = False
    ry.allow_unicode = True

    for tbl in tables:
        coefs = load_coefs(h5_path, tbl)
        if coefs is None:
            print(f"  [SKIP] {tbl}: could not load coefficients")
            continue

        # Convert numpy arrays/scalars to Python types for YAML serialization
        coefs_serializable = serialize_for_yaml(coefs)

        # Wrap under 'input.coefs' key to match config structure
        doc = {"input": {"coefs": coefs_serializable}}

        out_file = output_dir / f"{tbl}.yaml"
        with out_file.open(encoding="utf8", mode="w") as fp:
            ry.dump(doc, fp)

        print(f"  {tbl} -> {out_file.name}")

    print(f"\nExported {len(tables)} coefficient file(s) to {output_dir}")

def serialize_for_yaml(coefs):
    coefs_serializable = {}
    for k, v in coefs.items():
        if k == "dates":
            coefs_serializable[k] = {
                    dk: str(dv) for dk, dv in v.items()
                }
        elif isinstance(v, np.ndarray):
            coefs_serializable[k] = v.tolist()
        elif isinstance(v, np.generic):  # numpy scalar (np.str_, np.float64, …)
            coefs_serializable[k] = v.item()
        elif isinstance(v, str):
            coefs_serializable[k] = v
        elif hasattr(v, "tolist"):
            coefs_serializable[k] = v.tolist()
        else:
            coefs_serializable[k] = v
    return coefs_serializable


def main():
    parser = argparse.ArgumentParser(
        description="Export inclinometer coefficients from HDF5 to YAML"
    )
    parser.add_argument(
        "h5_path",
        nargs="?",
        default=r"C:\Work\Python\AB_SIO_RAS\tcm\tcm\cfg\coef\calibration.h5",
        help="Path to HDF5 file with coefficients",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=r"C:\Work\Python\AB_SIO_RAS\tcm\tcm\cfg\coef",  # \yaml_export
        help="Output directory for YAML files",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    output_dir = Path(args.output_dir)

    if not h5_path.exists():
        print(f"Error: HDF5 file not found: {h5_path}")
        sys.exit(1)

    print(f"HDF5: {h5_path}")
    print(f"Output: {output_dir}")
    export_coefs(h5_path, output_dir)


if __name__ == "__main__":
    main()