h5toGrid workflow

 - "csv2h5 -cfg.ini": convert all tabular data to Pandas HDF5 Store. (here and
 below cfg.ini means use of specific (different) configuration file for each source data and program).
 - "h5toGpx -cfg.ini": extract navigation data at time station starts to GPX waypoints
 - change starts of sections and excluded stations with specified symbol
 using MapSource or other GPX waypoints editor
 and save result as new GPX-sections file
 - todo: calc preferred section directions and save updated gpx file with this indication
 - "gpx2h5 -cfg.ini": save GPX-sections to sections table in Pandas HDF5 Store
 - prepare Views pattern for extract useful data on section
 and preferably control display
 - "grid2d_vsz": create similar Views files based on sections table and they
 output to create Surfer grids, also creates edges blank files for depth and
 top an bot edges, log useful info (including section titles) to &grid2d_vsz.log


Other utilites:
 viewsPropagate - change data for Views pattern and export images
 viewsPProc_CTDends - load filtered CTD data from Veusz sections, add nav.
from Veusz data source store, query specified data and save to csv.
 grid3d_vsz - create 2D grids in xy-planes
 h5_copyto - open, modify and save all taables in Pandas HDF5 Store (see also h5reconfig)

 CTD_calc - calculate parameters using hdf5 data:
 
 
 Notes:
 
 To use other navigation formats (for example NMEA) use GPSBabel to convert it to GPX first.

