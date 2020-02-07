## Plotting - 191101incl_load.py
contains steps:
1. _csv2h5_ - Converting (interval) data to hdf5 format with automatic time correction: improved resolution compared to the original 1s at a specified frequency to create an increasing sequence (from which you can build graphs with maximum resolution - relevant for small intervals, as well as for some processing algorithms).
1a. h5copy_coef - copy coefficients to the created hdf along the data path of the corresponding device
2. Calculate velocity and average
3. Calculate spectrograms.
4. _veuszPropagate_ - make graphics according to a given Veusz template
 
## Calibration of inclinometers - 190901incl_calibr.py
_incl_calibr_: According to calibration data at the stand:
1. Soft iron
The obtained coefficients are applied to the data in the pool for further adjustment of the coefficients.

_h5from_veusz_coef_: Processing data from the pool:
2. Zero speed calibration (vertical position)
3. Zero calibration of the direction of speed (north, hard iron)

## Construction of a set of the same type of graphs (including long displayed by a computer)
1. Prepare the Veusz template manually. To speed up:
    - load a small part / thinned part
    - delete calculations: replace the calculation formulas for the necessary data to display them with the result ("unlink" menu), the rest - delete. After that, it is better to save in vszh5, then configure the graphics.
2. Delete all data and save (graphs) in "~no_data~.vsz".
3. In text mode, replace all graphs from a large file with graphs from "~no_data~.vsz". Save the result as a template for the plotting program.
4. Run the program to build the remaining parts according to the template