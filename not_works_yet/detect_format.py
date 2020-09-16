#!/usr/bin/env python
# from messytables.dateparser import is_date
# import json
import datetime
import re
from os import path as os_path

import numpy as np
from messytables import CSVTableSet, any_tableset, type_guess, headers_guess, headers_processor, offset_processor, types

from to_pandas_hdf5.csv2h5 import init_input_cols
# DateSense.detect_format(["15 Dec 2014", "9 Jan 2015"])
# rule_pattern_hms = DateSense.DSPatternRule( (('%H','%I'),':','%M',':','%S'), posscore = 3 )
# DateSense.detect_format( dates, rules )
from utils2init import ini2dict, init_file_names, Ex_nothing_done, set_field_if_no, standard_error_info

time_regex = re.compile(r'''^(?P<H>\d{1,2}):(?P<M>\d{2}):(?P<S>\d{2})\S*''')


def is_date_format(string):
    return (re.match(r'''.*[dmbyY]+.*''', string))


class TimeType(types.TimeType):
    # my TimeType '12:30:40'
    def __init__(self, format='%H:%M:%S'):
        self.format = format

    def cast(self, value):
        if isinstance(value, self.result_type):
            return value
        if value in ('', None):
            return None
        match = time_regex.match(value)
        if match:
            hour, minute, second = (int(m) for m in match.groups())
            # = match(value[0:2]) #corrected by substucting 2
            # = int(value[3:5])
            # = int(value[6:8])
        if hour < 24:
            return datetime.time(hour, minute, second)
        else:
            return datetime.timedelta(hours=hour,
                                      minutes=minute,
                                      seconds=second)

    def __repr__(self):
        return "Time({})".format(self.format)


def parse_csv(filename, cfg_in):
    """
    Guess csv structure

    :param filename:
    :param cfg_in:
    :param known_structure: list of strings formats in order of columns, from start
    but may be not all (next is auto treeted)
    :return: lst_types, offset, headers 


    * quotechar - specifies a one-character string to use as the
        quoting character.  It defaults to '"'.
    * delimiter - specifies a one-character string to use as the
        field separator.  It defaults to ','.
    * skipinitialspace - specifies how to interpret whitespace which
        immediately follows a delimiter.  It defaults to False, which
        means that whitespace immediately following a delimiter is part
        of the following field.
    * lineterminator -  specifies the character sequence which should
        terminate rows.
    * quoting - controls when quotes should be generated by the writer.
        It can take on any of the following module constants:

        csv.QUOTE_MINIMAL means only when required, for example, when a
            field contains either the quotechar or the delimiter
        csv.QUOTE_ALL means that quotes are always placed around fields.
        csv.QUOTE_NONNUMERIC means that quotes are always placed around
            fields which do not parse as integers or floating point
            numbers.
        csv.QUOTE_NONE means that quotes are never placed around fields.
    * escapechar - specifies a one-character string used to escape
        the delimiter when quoting is set to QUOTE_NONE.
    * doublequote - controls the handling of quotes inside fields.  When
        True, two consecutive quotes are interpreted as one during read,
        and when writing, each quote character embedded in the data is
        written as two quotes
    Example:
    parse_csv(filename, ['%H:%M:%S'])
    """
    set_field_if_no(cfg_in, 'types', [])
    set_field_if_no(cfg_in, 'delimiter')
    with open(filename, 'rb') as fh:
        ext = os_path.splitext(filename)[1]
        # Load a file object:
        try:
            # If you are sure that file is csv use CSVTableSet(fh)
            from magic import MagicException  # because any_tableset uses libmagic
            table_set = any_tableset(fh, mimetype=None, extension=ext,
                                     delimiter=cfg_in['delimiter'])
        except (ImportError, MagicException) as e:
            print('There are error ', standard_error_info(e),
                  '\n=> Loading file as csv without trying other formats')
            table_set = CSVTableSet(fh, delimiter=cfg_in['delimiter'])

        # A table set is a collection of tables:
        row_set = table_set.tables[0]
        # A row set is an iterator over the table, but it can only
        # be run once. To peek, a sample is provided:

        # guess header names and the offset of the header:
        offset, headers = headers_guess(row_set.sample)  # tolerance=1
        row_set.register_processor(headers_processor(headers))
        # add one to begin with content, not the header:
        row_set.register_processor(offset_processor(offset + 1))
        # guess column types:
        lst_types = type_guess(row_set.sample, strict=True)
        row_sample = next(row_set.sample)

        # check not detected types
        def formats2types(formats_str):
            for f in formats_str:
                if f:
                    if is_date_format(f):
                        yield (types.DateType(f))
                    else:
                        yield (TimeType())
                else:
                    yield (None)

        known_types = formats2types(cfg_in['types'])

        for n, (t, s, kt) in enumerate(zip(lst_types, row_sample, known_types)):
            if t.result_type == types.StringType.result_type:
                # not auto detected? -> check known_types
                if kt.test(s.value):
                    lst_types[n] = kt  # t= kt
                else:  # known_types fits element
                    print("col'" 's#{:d} value "{}" type not match provided type of {}'
                          .format(n, s.value, type(kt)))
                    # kt = types.DateType('mm/dd/yyyy')
                    # kt.test('0'+s.value)
                    # detect?
            else:
                pass
        # not works for time type:
        # print(jts.headers_and_typed_as_jts(headers,
        #       list(map(jts.celltype_as_string, lst_types))).as_json())
        return lst_types, offset, headers

        # # and tell the row set to apply these types to
        # # each row when traversing the iterator:
        # row_set.register_processor(types_processor(types))
    #
    # # now run some operation on the data:
    # for row in row_set:
    #     do_something(row)

    # import csv
    # with open(filename, 'rb') as csvfile:
    #     temp_lines = csvfile.readline() + '\n' + csvfile.readline()
    #     dialect = csv.Sniffer().sniff(temp_lines, delimiters=',|')
    #
    #     #dialect = csv.Sniffer().sniff(csvfile.read(1024), delimiters=";,")
    #     #csvfile.seek(0)
    #     #reader = csv.reader(csvfile, dialect)
    # return dialect


def parse(filename):
    import csv
    with open(filename, 'rb') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(), delimiters=';,')
        csvfile.seek(0)
        reader = csv.DictReader(csvfile, dialect=dialect)

        for line in reader:
            print(line['ReleveAnnee'])


# import re
# import numpy as np
#
# fieldFilter = re.compile(r'^"?([^"]*)"?$')
# def filterTheField(s):
#     m = fieldFilter.match(s.strip())
#     if m:
#         return float(m.group(1))
#     else:
#         return 0.0 # or whatever default
#
# #str.replace('"', '') method should perform noticeably faster
#
# # NumPy have to know the number of columns, since
# #you can not specify a default converter for all columns.
# convs = dict((col, filterTheField) for col in range(numColumns))
# data = np.genfromtxt(csvfile, dtype=None, delimiter=',', names=True,
#     converters=convs)


''' date '''


def find_date(str_date, format_found):
    from datetime import datetime
    try:
        x = datetime.strptime(str_date, format_found)
    except ValueError:
        # column is not valid.
        pass

    # dateutils.parser.parse
    from dateutil.parser import parse
    parse('April 12, 2013')  # datetime.datetime(2013, 4, 12, 0, 0)
    parse('04/12/13')  # datetime.datetime(2013, 4, 12, 0, 0)

    # deelen = pandas.read_csv('Deelen2.csv', parse_dates = [[0,1]], header = 0,   index_col = 0, delimiter=';', low_memory=False)
    import io
    import pandas as pd
    t = """
    date;time;DD;FH;FF;FX;T;
    20110101;1;240;30;30;40;15;
    20110101;2;250;30;40;60;18;
    20110101;3;250;40;40;70;21;
    20110101;4;250;40;30;60;20;
    20110101;5;250;40;40;60;21;
    """
    df = pd.read_csv(io.StringIO(t), sep=';', dtype=({'date': str}))
    df['date_time'] = pd.to_datetime(df['date']) + pd.TimedeltaIndex(df['time'], unit='H')


# ##############################################################################
# ___________________________________________________________________________
if __name__ == '__main__':
    #    unittest.main()
    import argparse

    parser = argparse.ArgumentParser(description='Detect format of CSV and other'
                                                 'files ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='If use special characters then insert arguments in quotes',
                                     )
    parser.add_argument('--version', action='version', version='%(prog)s '
                                                               'version 0.0.1 - (c) 2016 Andrey Korzh <ao.korzh@gmail.com>')  # sourceConvert

    parser.add_argument('cfgFile', nargs='?', type=str, default='sourceConvert.ini',
                        help='Path to confiuration *.ini file with all parameters. '
                             'Next parameters here overwrites them')
    info_default_path = '[in] path from *.ini if specified'
    parser.add_argument('path', nargs='?', type=str, default=info_default_path,
                        help='Path to source file(s) to parse')
    parser.add_argument('out', nargs='?', type=str, default='./<filename>.h5/gpx',
                        help='''Output .h5/table path.
If "<filename>" found it will be sabstituted with [1st file name]+, if "<dir>" -
with last ancestor directory name. "<filename>" string
will be sabstituted with correspondng input file names.
''')
    parser.add_argument('-verbose', nargs=1, type=str, default=['INFO'],
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        help='Verbosity of messages in log file')

    args = parser.parse_args()
    args.verbose = args.verbose[0]
    try:
        cfg = ini2dict(args.cfgFile)
        cfg['in'] = init_file_names(cfg['in'])
    except (IOError, Ex_nothing_done) as e:
        cfg = {'in': {'paths': [args.path],
                      'nfiles': 1}}  # one file
        if not os_path.isfile(args.path):
            print('\n==> '.join([s for s in e.args if isinstance(s, str)]))  # e.message
            raise (e)

    ############################################################################
    # set_field_if_no(cfg['in'],'types',None)
    # set_field_if_no(cfg['skiprows'],'types')
    if cfg['in']['nfiles']:

        # ## Main circle ############################################################
        for ifile, nameFull in enumerate(cfg['in']['paths'], start=1):
            nameFE = os_path.basename(nameFull)

            # known_structure ['%m/%d/%Y', '%H:%M:%S']
            lst_types_cur, skiprows_cur, headers = parse_csv(nameFull, cfg['in'])

            # Check obtained format is the same
            if cfg['in']['types'] != lst_types_cur:
                if not len(cfg['in']['types']):
                    cfg['in']['types'] = lst_types_cur
                else:
                    print('file {} has another format!'.format(nameFE))

            if cfg['in']['skiprows'] != skiprows_cur:
                if cfg['in']['skiprows'] is None:
                    cfg['in']['skiprows'] = skiprows_cur
                else:
                    print('file {} has another format!'.format(nameFE))

            # Convert to numpy types
            set_field_if_no(cfg['in'], 'max_text_width', 2000)  # big width for 2D blocks
            types2numpy = {t.result_type: np_type for t, np_type in zip(types.TYPES, '')}

            #     {
            #     types.StringType.result_type: '|S{:.0f}'.format(cfg['in']['max_text_width'])
            # }
            cfg['in']['dtype'] = np.array([np.float64] * len(cfg['in']['types']))
            for k, typ in enumerate(cfg['in']['types']):
                if typ.result_type == types.StringType.result_type:
                    pass
                cfg['in']['dtype'][k] = types2numpy[typ.result_type]

            # Prepare cpecific format loading
            cfg['in'] = init_input_cols(cfg['in'])
            cfg['out']['names'] = np.array(cfg['in']['dtype'].names
                                                    )[cfg['in']['cols_loaded_save_b']]
            cfg['out']['formats'] = [cfg['in']['dtype'].fields[
                                                  n][0] for n
                                              in cfg['out']['names']]
            cfg['out']['dtype'] = np. \
                dtype({
                'formats': cfg['out']['formats'],
                'names': cfg['out']['names']})
            # Load data

# '4	15,10,3,19,34,1,64	44.45140000	49.92563500'
