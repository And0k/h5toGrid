#!/usr/bin/env python
# coding:utf-8


"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>
  Purpose: Convert XML files which contains backuped SMS messages from GPS tracker to GPX tracks
  Created: 26.02.2016
"""
import re
import logging
from datetime import datetime as datetime
import xml.etree.ElementTree as ET
import gpxpy.gpx as GPX
# from codecs import

from utils2init import Ex_nothing_done, cfg_from_args, my_argparser_common_part, this_prog_basename, init_logging, standard_error_info

if __name__ == '__main__':
    l = None  # see main(): l = init_logging('', cfg['program']['log'], cfg['program']['verbose'])
else:
    l = logging.getLogger(__name__)
version = '0.0.1'


def my_argparser():
    """
    Configuration parser
    - add here common options for different inputs
    - add help strings for them
    :return p: configargparse object of parameters
    All p argumets are of type str (default for add_argument...), because of
    custom postprocessing based of args names in ini2dict
    """

    p = my_argparser_common_part({'description': 'SMS2GPX version {}'.format(version) + """
----------------------------
Convert SMS *.xml to *.gpx
----------------------------"""}, version)  # config_file_paths=r'SMS2GPX.ini'

    s = p.add_argument_group('in', 'XML files')
    s.add('--path', help='Path to XML or directory with XML files to parse')
    s.add('--dt_from_utc_hours', default='0',
             help='add this correction to loading datetime data. Can use other suffixes instead of "hours"')
    s.add('--contacts_names',
             help='list of contacts names to use like "tracker 3, Трекер 0, Трекер 1"')

    s = p.add_argument_group('out', 'XML files')
    s.add('--out.path', default='./<filename>.gpx',
              help='''Output dir/path.
    Join data from all found input files to single output if extension provided. If
    "<filename>" found it will be sabstituted with [1st file name]+, if "<dir>" -
    with last directory name.
    Else, if no extension provided then ".gpx" will be used, "<filename>" string
    will be sabstituted with correspondng input file names.
    ''')
    s.add('--dt_between_track_segments_hours', default='99999',
              help='''dt between track segments. also can try other valid time interval suffixes - all different suffix options will be summed''')

    s = p.add_argument_group('process', 'calculation parameters')
    # s.add_argument('--b_missed_coord_to_zeros',
    #                     help='out all points even if no coordinates, but replace them to zeros')
    s.add_argument('--min_date',
                        help='UTC, not output data with < min_date (if date is smaller then treat it as bad, so tries get from stamp if b_all_time_from_stamp is False. If it smaller too then data discard, format like in 13.05.2017 09:00:00')

    s = p.add_argument_group('program', 'Program behaviour')
    s.add_argument('--log',
                           help='write log if path to existed file is specified')
    return p


def parse_smses(cfg):
    """
    Parse valid messages:
old tracker:
lat:54.719937 long:20.525533 speed:0.00 dir:0.0\nT:21/02/16 01:45\nhttp://maps.google.com/maps?f=q&q=54.719937,20.525533&z=16
new tracker:
lat:54.735578\nlong:20.544967\nspeed:0.07 \nT:19/10/13 05:03\nbat:100%\nhttp://maps.google.com/maps?f=q&q=54.735578,20.544967&z=16

(special_symbol (&#10;|&amp;) parsing not need because replaced by ET.parse)
    :param cfg:
    :return:
    """

    r_coord_val = '\d{1,3}\.\d{1,8}'
    r_lat = f'(?: ?lat:)(?P<Lat>{r_coord_val})'
    r_lon = f'(?:long:)(?P<Lon>{r_coord_val})'
    r_speed = '(?:speed:)(?P<Speed>\d{1,3}\.\d{1,3})'
    r_time = '(?:\nT:)(?P<Time>\d\d/\d\d/\d\d \d\d\:\d\d)'  # search time string '24/02/16 20:40'

    re_msg = []
    for b_new in [False, True]:
        if b_new:
            r_dir = ' '
            r_sep = '\n'
            r_power = '(?:\nbat:)(?P<Power>\d{1,3})(?:%)'
            r_z = '((?:&z=)(?P<Z>\d{1,3})|)'
        else:
            r_dir = '(?: dir:)(?P<Dir>\d{1,3})'
            r_sep = ' '
            r_power = '((?:&z=)(?P<Z>\d{1,3})|)'
            r_z = ''
        re_msg.append(re.compile(f'{r_lat}{r_sep}{r_lon}{r_sep}{r_speed}{r_dir}{r_time}{r_power}.*{r_z}'))

    bWriteComments = 'b_write_comments' in cfg['process'] and cfg['process']['b_write_comments']
    bWriteWithoutCoord = 'b_write_without_coord' in cfg['process'] and cfg['process']['b_write_without_coord']
    bAllTimeFromStamp = 'b_all_time_from_stamp' in cfg['process'] and cfg['process']['b_all_time_from_stamp']
    # min_str_date= cfg['process']['min_str_date'] if 'min_str_date' in cfg['process'] else datetime.strptime('01.01.2010', '%d.%m.%Y')
    min_date = cfg['process']['min_date'] if 'min_date' in cfg['process'] else datetime.strptime('01.01.2010',
                                                                                                 '%d.%m.%Y')
    re_msgBad = re.compile(f'^(?P<Comment>[^T]*){r_time}.*')

    #     re_msg= re.compile('^(?P<Comment>[^\:]*?)(?: ?lat[:])(?P<Lat>\d{1,3}\.\d{1,8}) (?:long[:])(?P<Lon>\d{1,3}\.\d{1,8}) \
    # (?:speed[:])(?P<Speed>\d{1,3}\.\d{1,3}) (?:dir[:])(?P<Dir>\d{1,3}).*' + r_time)
    kmh2ms = 1000.0 / 3600.0

    getFromStamp = lambda n: datetime.utcfromtimestamp(int(n.attrib['date'] if \
                                                               len(n.attrib['date_sent']) <= 1 else n.attrib[
        'date_sent']) / 1000.0)  # 'date2'

    def gettime(node=None, m_groupTime=None, b_year_first=None):
        """

        :param node:
        :param m_groupTime:
        :param b_year_first: if None act like bAllTimeFromStamp == True
        :return:
        """
        if bAllTimeFromStamp or (b_year_first is None):
            time_b = getFromStamp(node)
            if time_b < min_date:
                return None
        else:
            try:
                time_b = datetime.strptime(
                    m_groupTime,
                    r'%y/%m/%d %H:%M' if b_year_first else r'%d/%m/%y %H:%M'
                    ) - cfg['in']['dt_from_utc']
                if time_b < min_date: raise ValueError
            except ValueError:
                time_b = getFromStamp(node)
                if time_b > min_date:
                    print('bad time: "' + m_groupTime + '" replaced by stamp')
                else:
                    return None
        return time_b

    tree = ET.parse(cfg['in']['path'])
    root = tree.getroot()

    # Create tracks in our GPX
    gpx = GPX.GPX()
    gpx_track = {}
    # gpx.tracks= [GPX.GPXTrack(name = c) for c in contacts]
    # gpx_track= dict(zip(contacts, gpx.tracks))
    for contact_name in cfg['in']['contacts']:
        contact_name_d = contact_name  # .decode('cp1251') #allow Russian letters in contact names
        # try:
        #     print('contact name: ', contact_name_d)
        # except UnicodeEncodeError:
        #     contact_name_d= contact_name.decode('utf-8') #.encode('cp1251')
        time_b_prev = datetime.now()  # or any big value
        gpx_track[contact_name] = GPX.GPXTrack(name=contact_name_d)
        # if gpx_track[contact_name].number: #.get_points_no():
        print('contact name: ', contact_name_d)
        # else:
        # continue
        gpx.tracks.append(gpx_track[contact_name])
        # Create segment in this GPX track:
        gpx_segment = GPX.GPXTrackSegment()
        gpx_track[contact_name].segments.append(gpx_segment)

        old_or_new = [False, True]
        b_year_first = None
        for neighbor in root.findall("./sms/[@contact_name='" + contact_name_d + "'][@type='1']"):  # received SMS
            # Parse messages:
            body = neighbor.attrib['body']
            for b_new in old_or_new:
                m = re_msg[b_new].search(body)
                if m is not None:
                    old_or_new = [b_new]  # fix neweness for contact for speedup
                    b_year_first = b_new
                    break
            if (m is not None) and (m.group(1) is not None):  # valid => create points:
                Comment = m.group('Comment') if bWriteComments else None
                time_b = gettime(neighbor, m.group('Time'), b_year_first=b_year_first)
                if not time_b:
                    continue
                # Add segment for big time intervals:
                if time_b - time_b_prev > cfg['out']['dt_between_track_segments']:
                    gpx_segment = GPX.GPXTrackSegment()
                    gpx_track[contact_name].segments.append(gpx_segment)
                time_b_prev = time_b
                if m.group('Speed') != '0.00':
                    speed_b = round(float(m.group('Speed')) * kmh2ms, 5)
                else:
                    speed_b = '0'  # work around: gpxpy not writes 0 if float
                gpx_point = GPX.GPXTrackPoint(
                    latitude=m.group('Lat'), longitude=m.group('Lon'),
                    elevation=m.group('Z') if 'Z' in m.re.groupindex.keys() else None,
                    time=time_b, speed=speed_b, comment=Comment)
                if 'Dir' in m.re.groupindex.keys():
                    gpx_point.course = m.group('Dir')  # where to write dir?
            elif bWriteWithoutCoord:  # invalid => messages
                m = re_msgBad.match(body)
                if (m is not None) and (m.group(1) is not None):  # valid time=> create comment
                    Comment = m.group('Comment') if bWriteComments else None
                    time_b = gettime(neighbor, m.group('Time'), b_year_first=b_year_first)
                    if not time_b:
                        continue
                    gpx_point = GPX.GPXTrackPoint(time=time_b, comment=Comment)
                else:  # get Time from message receive time stamp
                    Comment = body if bWriteComments else None  # Consider full message as comment
                    try:
                        time_b = getFromStamp(neighbor)
                        if time_b < min_date:
                            continue  # raise ValueError
                    except:
                        time_b = None
                        print('can not find time in message and time stamp')
                        continue
                    gpx_point = GPX.GPXTrackPoint(time=time_b)
            else:
                continue
            gpx_segment.points.append(gpx_point)
        gpx.description = contact_name_d
        gpx.author_email = 'andrey.korzh@atlantic.ocean.ru'
    return gpx


# ##############################################################################
def main(new_arg=None, **kwargs):
    global l

    if __package__ is None:
        from sys import path as sys_path
        from os import path as  os_path
        sys_path.append(os_path.dirname(os_path.dirname(os_path.abspath(__file__))))

    from utils2init import prep

    cfg = cfg_from_args(my_argparser(), new_arg, **kwargs)

    # Input files
    default_input_filemask = '*.xml'
    inD, namesFE, nFiles, outD, outF, outE, bWrite2dir, msgFile = prep(
        {'path': cfg['in']['path'], 'out_path': cfg['out']['path']},
        default_input_filemask)

    l = init_logging('', cfg['program']['log'], cfg['program']['verbose'])
    l.warning('\n' + this_prog_basename(__file__) + ' started. ')

    l.warning(msgFile)
    # set_field_if_no(cfg['out'], 'dt_between_track_segments', 99999)

    gpx = parse_smses(cfg)
    try:
        f = open(cfg['in']['path'].with_suffix('.gpx'), 'w')
        bMissedCoordTo0 = 'b_missed_coord_to_zeros' in cfg['process'] and cfg['process']['b_missed_coord_to_zeros']
        if bMissedCoordTo0:
            for p in gpx.walk(only_points=True):
                if p.latitude is None or p.longitude is None:
                    p.latitude = '0'  # float('nan') #0
                    p.longitude = '0'  # float('nan') #0
                # if p_prev==p:
                # p.delete
                # p_prev= p

        # gpx.add_missing_data() #remove_empty()
        f.write(gpx.to_xml())
        print('ok')
    except Ex_nothing_done as e:
        print(e.message)
    except Exception as e:
        msg_option = f'The end. There are error {standard_error_info(e)}'
        print(msg_option)
        try:
            err_msg = e.msg
            l.error(' '.join([err_msg, msg_option]))
        except AttributeError:
            l.error(msg_option)
    finally:
        f.close()
        try:
            # if not bWrite2dir:
            #     fp_out.close()
            # l.handlers[0].flush()
            logging.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()

'''
TRASH

#to reformat time string:
#'24/02/16 20:40' to GPX.DATE_FORMAT='%Y-%m-%dT%H:%M:%SZ'
re_msgBad= re.compile('(\d\d)/(\d\d)/(\d\d) (\d\d\:\d\d)') #useful fields in parenthesies
bodyTimeStr, ok= re_msgBad.subn(r'\3-\2-\1T\4:00Z', m.group('Time'), count=1)

datetime.fromtimestamp(time_b/1000, pytz.timezone('UTC'))
'''
