This repository contains as good files as bunch of garbage. Mainly the code here is used to convert format of <a href="https://en.wikipedia.org/wiki/CTD_(instrument)">CTD</a> data and <a href="https://jor.ocean.ru/index.php/jor/article/download/369/153">inclinometer</a> data.

## Requirements
After having many problems with pip/pipenv on windows I use conda for now. [There](py3.10x64h5togrid.yml) is my [Conda environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#id16) `py3.10x64h5togrid.yml`. To install environment with Python 10 with latest versions of all needed packages use:
```cmd
conda env create --force --file py3.10x64h5togrid.yml
```

## Workflow to calculate grids from CTD data 
*Examples of workflow can be found in ``scripts`` directory*. They usually includs steps:

 - ``csv2h5``: convert all tabular data to PyTables HDF5 Store which is used by [pandas](https://pandas.pydata.org) library and can be loaded in [Veusz](https://github.com/veusz/veusz).
 - ``h5toGpx``: extract navigation data at time station starts to GPX waypoints
 - create new GPX-sections file (using Garmin MapSource or other GPX waypoints editor) that contains or:
    - routes over stations waypoints (method allows adjust direction of each section) or
    - same waypoints but breaks (i.e. starts) of sections and excluded stations marked by special symbols
 - ``gpx2h5``: save GPX-sections to sections table in PyTables HDF5 Store
 - prepare Veusz pattern for extract useful data on section
 and preferably control display
 - ``grid2d_vsz``: create similar Views files based on sections table and their data
 output to create [Surfer](https://www.goldensoftware.com/products/surfer) grids, also creates edges blank files for depth and
 top an bot edges, log useful info (including section titles) to &grid2d_vsz.log


Other utilities:
 - ``viewsPropagate`` - change data for Views pattern and export images
 - ``viewsPProc_CTDends`` - load filtered CTD data from Veusz sections, add nav.
from Veusz data source store, query specified data and save to csv.
 - ``grid3d_vsz`` - create 2D grids in xy-planes
 - ``h5_copyto`` - open, modify and save all taables in Pandas HDF5 Store (see also h5reconfig)

 - ``CTD_calc`` - calculate CTD parameters based on input CTD parameters in hdf5 format.
 
 
### Notes:
Almost all programs can be executed in shell or as imported python functions with same arguments. Programs has named command line arguments that start with '--' (eg. ``--path C:\data.txt``) that can also be set in a config file. Programs has also first optional positional argument of path of configuration file (ini or yaml) where parameters has same names as arguments, but also section (group of arguments). If an argument is specified in more than one place, then command line values override config file values which override defaults. A description of all the parameters can be obtained by calling programs from the command line with the --help switch (instead of the path to the parameter file).

 
 If input navigation format is not GPX (for example NMEA) then convert it (I use [GPSBabel](https://www.gpsbabel.org/)).



I usually use this directions for calculate distance in sections:
 - Gdansk Deep at right in Baltic
 - Nord always at left
 