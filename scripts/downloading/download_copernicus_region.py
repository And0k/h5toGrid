import copernicusmarine as cm
import requests
# environment to run this script:
# conda create -n py12copernicus python=3.12 copernicusmarine jupyter_console -c conda-forge
# conda activate py12copernicus && jupyter console
# %run C:\Work\Python\AB_SIO_RAS\h5toGrid\scripts\download_copernicus_region.py
# see also https://ipywidgets.readthedocs.io/en/stable/user_install.html

"""
Baltic Sea Wave Analysis and Forecast

Часовые данные с разрешением 2 × 2 km
Baltic Sea Wave Analysis and Forecast (BALTICSEA_ANALYSISFORECAST_WAV_003_010, 1 Oct 2021)
Baltic Sea Wave Hindcast (BALTICSEA_MULTIYEAR_WAV_003_015, 1 Jan 1980 to 1 Dec 2024)
Sea surface wave maximum height VCMX [m]
Sea surface wave significant height VHM0 [m]
Sea surface wave from direction VMDR [°]
Sea surface wave maximum crest height   VMXL [m]
Sea surface wave from direction at variance spectral density maximum    VPED [°]
Sea surface wave mean period from variance spectral density inverse frequency moment VTM10 [s]
Sea surface wave mean period from variance spectral density second frequency moment VTM02 [s]
Sea surface wave period at variance spectral density maximum    VTPK [s]

Sea surface wind wave significant height VHM0_WW [m]
Sea surface wind wave from direction VMDR_WW [°]
Sea surface wind wave mean period VTM01_WW [s],

copernicusmarine.subset(
    dataset_id="cmems_mod_bal_wav_anfc_PT1H-i",
    dataset_version="202311",
    variables=["VCMX", "VHM0", "VTM10", "VTPK", "VTM02"],
    minimum_longitude=9.013887405395508,
    maximum_longitude=30.207738876342773,
    minimum_latitude=53.0082893371582,
    maximum_latitude=65.90777587890625,
    start_datetime="2025-03-29T12:00:00",
    end_datetime="2025-03-29T12:00:00",
    coordinates_selection_method="strict-inside",
    disable_progress_bar=True,
)
copernicusmarine.subset(
    dataset_id="cmems_mod_bal_wav_my_PT1H-i",
    variables=["VCMX", "VHM0", "VPED", "VTM02", "VTM10", "VTPK", "VMDR_WW", "VMXL"],
    end_datetime="2024-12-01T00:00:00",
)



? Climatology, month of year: BALTICSEA_MULTIYEAR_WAV_003_015.mems_mod_bal_wav_my_2km-climatology_P1M-m
Variables
Sea surface wave significant wave height VHM0 [m]
Sea surface wave mean period from variance spectral density second frequency moment VTM02 [s]
copernicusmarine.subset(
    dataset_id="cmems_mod_bal_wav_my_2km-climatology_P1M-m",
    dataset_version="202411",
    variables=["VHM0", "VTM02"],
    minimum_longitude=9.013799667358398,
    maximum_longitude=30.20800018310547,
    minimum_latitude=53.0082893371582,
    maximum_latitude=65.90809631347656,
    start_datetime="2001-12-01T00:00:00",
    end_datetime="2001-12-01T00:00:00",
    coordinates_selection_method="strict-inside",
    disable_progress_bar=True,
)
"cmems_mod_bal_wav_my_2km-climatology_P1M-m": ["VHM0", "VTM02"]



# "cmems_mod_bal_phy_anfc_P1D-m",  # Baltic Sea Physics Analysis and Forecast (1 Nov 2021)

    dataset_id="cmems_mod_bal_phy_my_P1M-m",  # Monthly (for Daily use: cmems_mod_bal_phy_my_P1D-m)
    dataset_version="202303",
    variables=["bottomT", "so", "sob", "thetao", "uo", "vo"],
    minimum_longitude=9.041532516479492,
    maximum_longitude=30.20798683166504,
    minimum_latitude=53.00829315185547,
    maximum_latitude=65.89141845703125,
    start_datetime="2015-01-01T00:00:00",
    end_datetime="2023-12-01T00:00:00",
    minimum_depth=0.5016462206840515,
    maximum_depth=0.5016462206840515,
    coordinates_selection_method="strict-inside",
    disable_progress_bar=True,


copernicusmarine.subset(
    dataset_id="cmems_mod_bal_bgc_my_P1M-m",
    dataset_version="202303",
    variables=["chl", "nh4", "no3", "nppv", "o2", "o2b", "ph", "po4", "spco2", "zsd"],
    minimum_longitude=9.041532516479492,
    maximum_longitude=30.20798683166504,
    minimum_latitude=53.00829315185547,
    maximum_latitude=65.89141845703125,
    start_datetime="2023-12-01T00:00:00",
    end_datetime="2023-12-01T00:00:00",
    minimum_depth=0.5016462206840515,
    maximum_depth=0.5016462206840515,
    coordinates_selection_method="strict-inside",
    disable_progress_bar=True,
)


Alternatively, to remotely open the dataset (xarray/OPeNDAP-like), consult the `open_dataset` function.
"""

def extract_error_from_xml(xml_string):
    xml_string = e.doc
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


# Add new config (only first tuple is used)
dir_save, *date_range, min_lon, max_lon, min_lat, max_lat = (
    ## 150101_Mariculture
    (  # 1
        r"D:\WorkData\BalticSea\CMEMS\150101_Mariculture",
        "2015-01-01",
        "2025-03-23",
        19.611389,
        19.846389,
        55.133889,
        55.300833,
    ),
    (  # 2
        r"D:\WorkData\BalticSea\CMEMS\150101_Mariculture",
        "2015-01-01",
        "2025-03-23",
        20.125833,
        20.478333,
        55.084722,
        55.243056,
    ),
    (  # 3
        r"D:\WorkData\BalticSea\CMEMS\150101_Mariculture",
        "2015-01-01",
        "2025-03-23",
        19.304722, 19.596667,
        54.551389, 54.769444,
    ),
    ###
    (  # D:\Downloads\_send
        r"D:\WorkData\_model\Copernicus\240616_ABP56(t-chain)",
        "2024-06-25",
        "2024-09-05",
        17.5,
        21.5,
        54.25,
        56,
    ),
    (
        r"d:\workData\BalticSea\_other_data\_model\NEMO@CMEMS\section_z\231220_inflow",
        "2023-09-15",
        "2024-02-10",  # ("2023-12-02", "2024-01-31")?
        10.51,
        21.32,
        54.01,
        58.99,
    ),
)[0]
# Older:
# dir_save = r"d:\workData\BalticSea\_other_data\_model\NEMO@CMEMS\section_z\231220_inflow"
# date_range = ("2023-10-01", "2023-12-01") ("2023-09-15", "2023-09-30") # result will include these edges
# min_lon, max_lon, min_lat, max_lat = 10.51, 21.32, 54.01, 58.99

dataset_vars = {
    # "cmems_mod_bal_bgc_anfc_P1D-m": ["o2","o2b"],
    "cmems_mod_bal_phy_anfc_P1D-m": ("bottomT","thetao","so","sob","uo","vo","wo")
    # "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H": ("eastward_wind", "northward_wind")
    # "cmems_mod_bal_phy_anfc_P1D-m": ("thetao","so","uo","vo")  # for Masha
}

#%% Reorder and rename

function_args = {
    "minimum_longitude": min_lon,
    "maximum_longitude": max_lon,
    "minimum_latitude": min_lat,
    "maximum_latitude": max_lat,
    "start_datetime": date_range[0],
    "end_datetime": date_range[-1],
    "minimum_depth": 0.5,
    "maximum_depth": 125,  # 92,
    "output_directory": dir_save,
    "netcdf_compression_level": 9,  # 0-9
    # deprecated (but how to pevent asking?):
    "force_download": True,  # don't ask save each file
}
err = None
paths = []
for i, (dataset, vars) in enumerate(dataset_vars.items()):
    print(dataset, vars, sep=": ")
    params = {
        **function_args,
        "dataset_id": dataset,
        "variables": vars  # "({})".format(",".join('"{v}"' for v in vars))
    }
    # Exclude depth settings for "wind"
    if vars[0].endswith("_wind"):
        for param_name in ["minimum_depth", "maximum_depth"]:
            del params[param_name]

    try:
        p = cm.subset(**params)
    except requests.exceptions.JSONDecodeError as e:
        err = e

        if hasattr(e, 'doc'):
            extract_error_from_xml(xml_string=e.doc)

    else:
        paths.append(p)
# 89114609630 Tinkoff Сергей
if not err:
    print("Saved to", paths, "Ok>")


"""

Exception has occurred: SSLError
(note: full exception trace is shown but execution is paused at: <module>)
HTTPSConnectionPool
(host='s3.waw3-1.cloudferro.com', port=443):
Max retries exceeded with url: /mdl-metadata/mdsVersions.json?x-cop-client=copernicus-marine-toolbox&x-cop-client-version=1.3.5
(Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1000)')))
urllib3.exceptions.MaxRetryError: HTTPSConnectionPool
(host='s3.waw3-1.cloudferro.com', port=443): Max retries exceeded with url: /mdl-metadata/mdsVersions.json?x-cop-client=copernicus-marine-toolbox&x-cop-client-version=1.3.5
(Caused by  ')))
"""