from to_vaex_hdf5.autofon_coord import call_example, proc

if True:  # b_hydra:
    call_example()

    # from to_vaex_hdf5.autofon_coord import main_call
    # from pathlib import Path
    #
    # path_db = Path(r'd:\WorkData\BlackSea\210408_trackers\210408trackers.h5')
    # main_call([
    # ])

else:
    # without hydra still possible to run:
    proc()