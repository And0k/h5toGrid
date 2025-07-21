import pytest
import cfg_dataclasses

from inclinometer.incl_h5clc_hy import *

# @pytest.fixture()
# def autofon_dict():
#     """
#
#     3179282432 - будильники
#     :return:
#     """
#     with Path(r'data\tracker\autofon_coord_200msgs.json').open('r') as fp:
#         return json.load(fp)
#
#
# #@pytest.mark.parametrize('autofon_dict', )
# def test_autofon_df_from_dict(autofon_dict):
#     df = autofon_df_from_dict(autofon_dict,timedelta(0))
#     assert ['Lat',
#             'Lon',
#             'Speed',
#             'LGSM',
#             'HDOP',
#             'n_GPS',
#             'Temp',
#             'Course',
#             #'Height',
#             #'Acceleration'
#             ] == df.columns.to_list()
#     dtype = np.dtype
#     assert [dtype('float32'),
#             dtype('float32'),
#             dtype('float32'),
#             dtype('int8'),
#             dtype('float16'),
#             dtype('int8'),
#             dtype('int8'),
#             dtype('int8'),
#             #dtype('int8'),
#             #dtype('int8')
#             ] == df.dtypes.to_list()
#
#     assert len(df) == 200
#
#
# @pytest.mark.parametrize(
#     'file_raw_local', [
#         r'data\tracker\AB_SIO_RAS_tracker',
#         r'data\tracker\SPOT_ActivityReport.xlsx'
#         ])
# def test_file_raw_local(file_raw_local):
#     path_raw_local = Path(file_raw_local)
#     cfg_in = {'time_interval': ['2021-06-02T13:49', '2021-06-04T20:00']}  # UTC
#     time_interval = [pd.Timestamp(t, tz='utc') for t in cfg_in['time_interval']]
#
#     df = loading(
#         table='sp4',
#         path_raw_local=path_raw_local,
#         time_interval=time_interval,
#         dt_from_utc=timedelta(hours=2)
#         )
#     assert all(df.columns == ['Lat', 'Lon'])
#     assert df.shape[0] == 3


path_db = Path(
    r'C:\Work\Python\AB_SIO_RAS\h5toGrid\inclinometer\tests\data\inclinometer\210519incl.h5'
).absolute()


@pytest.mark.parametrize('return_', ['<end>', '<cfg_before_cycle>'])
def test_call_example_sp4(return_):

    db_path_in = str(path_db).replace('\\', '/')
    device = 'incl10'
    aggregate_period_s = (0, 600)

    sys_argv_save = sys.argv.copy()
    sys.argv = [__file__]  # config dir will be relative to this dir

    df = cfg_dataclasses.main_call([
        f'input.db_path="{db_path_in}"',
        # '++filter.time_bad_intervals=[2021-06-02T13:49, now]', # todo
        'input.tables=["incl.*"]',  # (','.join([f'"{d}"' for d in device]))
        f'out.db_path="{db_path_in}proc.h5"',
        # f'out.table=V_incl_bin{aggregate_period_s}s',
        'out.b_del_temp_db=True',
        'program.verbose=INFO',
        'program.dask_scheduler=synchronous',
        f'program.return_="{return_}"',
        f"out.aggregate_period={','.join(f'{a}s' for a in aggregate_period_s)}",
        '--multirun',
        '--config-path=tests/hydra_cfg',
        #'--config-dir=hydra_cfg'  # additional cfg dir
        ], fun=main)

    if return_ == '<cfg_before_cycle>':
        cfg = df
        # assert 'in' in cfg

    sys.argv = sys_argv_save