This repository contains as good files as bunch of garbage. Mainly the code here is used to convert format of <a href="https://en.wikipedia.org/wiki/CTD_(instrument)">CTD</a> data and <a href="https://jor.ocean.ru/index.php/jor/article/download/369/153">inclinometer</a> data.

#Requirements
After having many problems with pip/pipenv on windows I now use conda. There is my [Conda environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#id16) py3.7x64h5togrid_no_pip0.yml that I've used to install what I need.


#Workflow to calculate grids from CTD data 
*Examples of workflows can be found in ``scripts`` directory*. They usually includs steps:

 - ``csv2h5 -cfg.ini``: convert all tabular data to PyTables HDF5 Store wich is used by [pandas](https://pandas.pydata.org) library and may be loaded in [Veusz](https://github.com/veusz/veusz). (here and below cfg.ini means use of specific (usually different) configuration file for each source data and program).
 - ``h5toGpx -cfg.ini``: extract navigation data at time station starts to GPX waypoints
 - change starts of sections and excluded stations with specified symbol
 using Garmin MapSource or other GPX waypoints editor
 and save result as new GPX-sections file
 - todo: calc preferred section directions and save updated gpx file with this indication
 - ``gpx2h5 -cfg.ini``: save GPX-sections to sections table in PyTables HDF5 Store
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

 - ``CTD_calc`` - calculate parameters using hdf5 data:
 
 
###Notes:
 
 If input navigation format is not GPX (for example NMEA) then convert it (I use [GPSBabel](https://www.gpsbabel.org/)).



I usulally use this directions for calculate distance in sections:
 - Gdansk Deep at right in Baltic
 - Nord always at left
 