import copernicusmarine as cm
import requests
# uses File C:\Users\User\.copernicusmarine\.copernicusmarine-credentials
# environment to run this script:
# conda create -n py12copernicus python=3.12 copernicusmarine jupyter_console -c conda-forge
# conda activate py12copernicus && jupyter console

def extract_error_from_xml(xml_string):
    assert xml_string.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    import xml.etree.ElementTree as ET

    def print_xml_elements(element, indent=0):
        """print all xml elements"""
        # Print the tag and text of the current element
        print(' ' * indent + f"{element.tag}: {element.text}")
        # Recursively print all child elements
        for child in element:
            print_xml_elements(child, indent + 2)
    try:
        # Attempt to parse the XML string
        root = ET.fromstring(xml_string)
        print_xml_elements(root)
    except Exception as ee:
        # Handle parsing errors
        print(
            "Error returned as xml:",
            xml_string,
            ": XML Parsing Error:" if isinstance(ee, ET.ParseError) else "some XML parse error",
            ee
        )

### User's settings
dir_save = (
    # r"D:\WorkData\BalticSea\CMEMS\150101_Mariculture\2024-12-01 - 2025-03-25"
    r"B:\WorkData\BalticSea\240625_ABP56-incl,t-chain\meteo\CMEMS"
    # r"d:\workData\BalticSea\_other_data\_model\NEMO@CMEMS\section_z\231220_inflow\230901-240630_LatLon=(55.383,15.099)(time)"
)
points = (
    (55.26527, 19.67321),
    # (55.216528, 19.726026), (55.163611,	20.308813), (54.692459,	19.502652)  # Центры полигонов
    # (54.696, 19.244), (54.77, 18.922)
    # (54.744417, 19.5799),
    # (54.885266, 13.86073), (55.050, 13.674), (55.383, 15.099),
)

date_range = (
    # ("2024-11-21", "2025-03-31")  # to continue `wind_glo_phy` reanalysis (techn.can from 2024-11-21)
    # ("2024-01-01", "2025-03-31")  # to continue `phy`, `bgs` reanalysis
    # ("2023-12-01", "2025-03-31")  # to continue VHM0_WW"...
    # ("2015-01-01", "2024-12-01")  # will be to 31 Dec 2023 ("2024-01-01")
    # ("2024-06-25", "2024-09-05")  # 25.06.2024
    ("2024-06-25", "2025-04-20")
    # ("2023-07-01", "2024-07-05")
)
dataset_vars = {
    "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H": ["eastward_wind", "northward_wind"],
    ## ("2023-12-01", "2025-03-30"):
    # "cmems_mod_bal_wav_anfc_PT1H-i": [
    #     "VHM0_WW", "VMDR_WW", "VTM01_WW", "VCMX", "VHM0", "VPED", "VTM02", "VTM10", "VTPK", "VMXL"
    # ],
    # "cmems_mod_bal_phy_anfc_P1M-m": ["bottomT", "so", "sob", "thetao", "uo", "vo"],  # monthly
    # "cmems_mod_bal_bgc_anfc_P1M-m": ["o2", "o2b"],  # monthly
    # "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H": ["eastward_wind", "northward_wind"],  # ("2024-11-21", "2025-03-29")
    # "cmems_mod_bal_phy_anfc_P1D-m": ["bottomT", "so", "sob", "thetao", "uo", "vo"],  # daily
    # "cmems_mod_bal_phy_anfc_P1D-m": ["mlotst", "sob"],  # daily
    # "cmems_mod_bal_bgc_anfc_P1D-m": ["o2", "o2b"],  # daily
    # "cmems_mod_bal_bgc_anfc_P1D-m": ["nh4", "chl"],  # daily
    # ("2015-01-01", "2024-12-01"):
    # "cmems_mod_bal_wav_my_PT1H-i": [
    #     "VHM0_WW", "VMDR_WW", "VTM01_WW", "VCMX", "VHM0", "VPED", "VTM02", "VTM10", "VTPK", "VMXL"],
    # "cmems_mod_bal_phy_my_P1M-m": ["bottomT", "so", "sob", "thetao", "uo", "vo"],  # monthly
    # "cmems_mod_bal_bgc_my_P1M-m": ["o2", "o2b"],  # monthly
    # "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H": ["eastward_wind", "northward_wind"],
    # "cmems_mod_bal_phy_my_P1D-m": ["bottomT", "so", "sob", "thetao", "uo", "vo"],  # daily
    # "cmems_mod_bal_phy_my_P1D-m": ["mlotst", "sob"],  # daily
    # "cmems_mod_bal_bgc_my_P1D-m": ["o2", "o2b"],  # +["zsd", "nppv"]? daily
    # "cmems_mod_bal_bgc_my_P1D-m": ["nh4", "chl"],  # daily
    # "cmems_mod_bal_phy_anfc_PT15M-i": ["sla"],
    # "cmems_mod_bal_phy-ssh_anfc_detided_P1D-m": ["zos_detided"],
    # "cmems_mod_bal_phy_anfc_PT1H-i": ["mlotst", "sla"],
    # 'cmems_mod_bal_phy_anfc_PT1H-i':  ('thetao','so','sob','uo','vo','wo')  #('thetao','so')
    # 'cmems_mod_bal_bgc_anfc_P1D-m': ['o2b'],
    # 'cmems_mod_bal_phy_anfc_PT1H-i': ('bottomT', 'sob'),
    # 'cmems_mod_bal_phy_anfc_static': ['deptho']
    # 'cmems_mod_bal_phy_anfc_P1D-m': ('bottomT', 'sob')
}

function_args = {
    "start_datetime": date_range[0],
    "end_datetime": date_range[-1],
    "minimum_depth": 0.5,
    "maximum_depth": 100,  # 92, 125
    "output_directory": dir_save,
    "netcdf_compression_level": 9,  # 0-9
    # deprecated (but how to prevent asking?):
    "force_download": True,  # don't ask save each file
}

### Loop
err = None
paths = []
for i, (dataset, vars) in enumerate(dataset_vars.items(), start=1):
    params = {
        **function_args,
        "dataset_id": dataset,
        "variables": vars  # "({})".format(",".join('"{v}"' for v in vars))
    }
    if vars == 'deptho':
        params["force_dataset_part"] = "bathy"
        del params['start_datetime']
        del params['end_datetime']

    for ip, (lat, lon) in enumerate(points, start=1):
        params.update({
            "minimum_longitude": lon,
            "maximum_longitude": lon,
            "minimum_latitude": lat,
            "maximum_latitude": lat,
        })
        print(
            f"Dataset #{i}/{len(dataset_vars)} {dataset} ({vars}) at point #{ip}/{len(points)}"
        )
        try:
            path_saved = cm.subset(**params)
        except requests.exceptions.JSONDecodeError as e:
            err = e
            if hasattr(e, 'doc'):
                extract_error_from_xml(xml_string=e.doc)
            raise e from None
        # else:
        #     path_saved = cm.subset(**params)
        paths.append(path_saved)
if not err:
    print("Saved to", paths, "Ok>")

if False:  # if pytables installed
    with pd.HDFStore(r"d:\workData\BalticSea\_other_data\_model\NEMO@CMEMS\section_z\231220_inflow\230901-240630_LatLon=(55.383,15.099)(time)\230901.h5") as h:
        for df, (dataset, vars) in zip(results, dataset_vars.items()):
            df.to_hdf(
            h,
            key=dataset,
            append=True,
            data_columns=True,
            format="table",
            index=False
        )
