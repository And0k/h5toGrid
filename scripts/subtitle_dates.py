# Andrey Korzh, 27.01.2024
# 
import logging
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import nullcontext
from argparse import ArgumentParser, ArgumentError

l = logging.getLogger(__name__)

def gen_date_subtitles(start, end, dt, date_fmt='%d.%m.%Y', ddate=timedelta(days=1), file=None):
    date_start = datetime.fromisoformat(start)
    date_end = datetime.fromisoformat(end) + timedelta(days=1)
    i = 1
    date = date_start
    t = datetime(1, 1, 1)
    t_str = t.strftime('%T,%f')[:-3]  # '%H:%M:%S,%f'
    with (Path(file).open('w') if file else nullcontext()) as hf:
        while date < date_end:
            t_end = t + dt
            t_str_end = t_end.strftime('%T,%f')[:-3]
            (hf.write if file else print)(f"""{i}\n{t_str} --> {t_str_end}\n{date:{date_fmt}}\n""")
            i += 1
            date += ddate
            t = t_end
            t_str = t_str_end
    if file:
        print(f'Ok: {file} written')

if __name__ == '__main__':
    try:
        parser = ArgumentParser(description="Create subtitles text file with constant rate dates", exit_on_error=False)
        parser.add_argument('start', help="start date in ISO format")
        parser.add_argument('end', help="end date in ISO format")
        parser.add_argument('rate', help="output rate in milliseconds", type=int)
        parser.add_argument('file', help="output .srt file")
        args = parser.parse_args()
        
        gen_date_subtitles(args.start, args.end, dt=timedelta(milliseconds=args.rate), file=args.file)
    except SystemExit as e:  # ArgumentError
        l.error('ArgumentError. Running embedded command instead...')
        gen_date_subtitles(start='2023-12-01', end='2024-02-03', dt=timedelta(milliseconds=250),
            file=r'd:\workData\BalticSea\_other_data\_model\NEMO@CMEMS\section_z\231220_inflow\231220_inflow\vert_sec_Belt-Arkona,Oresund-Bornholm(V,T,S,O)\subtitles.srt'
        )
