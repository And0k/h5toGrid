# Andrey Korzh, 27.01.2024
# 
import logging
from datetime import datetime, timedelta

def gen_date_subtitles(start, end, dt, date_fmt='%d.%m.%Y', ddate=timedelta(days=1)):
    date_start = datetime.fromisoformat(start)
    date_end = datetime.fromisoformat(end)
    i = 1
    date = date_start
    t = datetime(1, 1, 1)
    t_str = t.strftime('%T,%f')[:-3]  # '%H:%M:%S,%f'
    while date < date_end:
        t_end = t + dt
        t_str_end = t_end.strftime('%T,%f')[:-3]
        print(f"""{i}
{t_str} --> {t_str_end}
{date:{date_fmt}}
""")
        i += 1
        date += ddate
        t = t_end
        t_str = t_str_end

if __name__ == '__main__':
    gen_date_subtitles(start='2023-12-01', end='2024-01-31', dt=timedelta(milliseconds=250))
