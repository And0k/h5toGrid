from .coefs import (
    coef_rotate,
    coefs_format_for_h5,
    get_coef_azimuth_shift,
    get_coefs,
    get_coefs_from_cfg,
    load_coefs,
    mag_dec
)
from .calc import (
    dekart2polar_df_uv,
    i_bursts_starts,
    f_linear_end,
    f_linear_k,
    fG,
    fIncl_deg2force,
    fIncl_rad2force,
    fVabs_from_force,
    fVabsMax0,
    out_velocity_cols,
    norm_field,
    polar2dekart,
    rep_if_bad,
    trigonometric_series_sum,
    v_abs_from_incl,
    v_trig,
)
