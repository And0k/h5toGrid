@echo off
set dir_scripts=%cd%
call conda.bat activate py3.11h5togrid
D: && cd D:\Work\_Python3\And0K\tcm || goto :error
@echo on

python -m incl_h5clc_hy ^
input.path='%dir_scripts%/_raw/230616-20@iP1-3.zip\INKL_P{number}.TXT' ^
out.raw_db_path='' ^
 || goto :error
@echo off
rem hydra.job_logging.formatters.colorlog.format='%%(cyan)s%%(asctime)s %%(blue)s%%(funcName)10s%%(cyan)s: %%(log_color)s%%(message)s%%(reset)s\t' ^
rem "hydra.job_logging.formatters.colorlog.format='%(cyan)s%(asctime)s %(blue)s%(funcName)10s%(cyan)s: %(log_color)s%(message)s%(reset)s\t'",  # \\< not supported
rem "+hydra/run/dir='outputs/${now:%Y%m%d_%H%M}'"
rem out.aggregate_period=[0s, 2s, 600s, 3600s, 7200s]
rem out.raw_db_path="{str_raw_db_path}"
rem 'program.verbose=INFO',
rem 'program.dask_scheduler=threads',
rem # set additional parameters in probes config directory defined in hydra.searchpath earlier
rem f'+probes={probes}',
rem #f'out=out0',
rem '--config-path=cfg_proc',  # Primary config module 'inclinometer.cfg_proc' not found.
rem '--config-dir=cfg_proc'  # additional cfg dir
rem 'input.min_date=2023-05-08T15:40',


call conda.bat deactivate
@echo off
cd "%dir_scripts%"
exit /b 0

@echo on
:error
@echo off
echo Failed with error #%errorlevel%.
cd "%dir_scripts%"
exit /b %errorlevel%
@echo on
