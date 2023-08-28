<?xml version="1.0" encoding="utf-8"?>
<!-- Created by Leo: http://leoeditor.com/leo_toc.html -->
<leo_file xmlns:leo="http://leoeditor.com/namespaces/leo-python-editor/1.1" >
<leo_header file_format="2" tnodes="0" max_tnode_index="0" clone_windows="0"/>
<globals body_outline_ratio="0.5" body_secondary_ratio="0.5">
	<global_window_position top="50" left="50" height="500" width="700"/>
	<global_log_window_position top="0" left="0" height="0" width="0"/>
</globals>
<preferences/>
<find_panel_settings/>
<vnodes>
<v t="And0K.20160516172816.2" a="E"><vh>@button SMS2GPX</vh>
<v t="And0K.20160516172920.1"><vh>@clean SMS2GPX.py</vh></v>
<v t="And0K.20160516173313.1" a="E"><vh>@auto SMS2GPX.ini</vh></v>
</v>
</vnodes>
<tnodes>
<t tx="And0K.20160516172816.2">@language python
c.k.simulateCommand('clear-log')
#c.k.SMS2GPX.py</t>
<t tx="And0K.20160516172920.1">@language python
#!/usr/bin/env python
#coding:utf-8

"""
  Author:  Andrey Korzh --&lt;korzh@nextmail.ru&gt;
  Purpose: Convert XML files wich contains backuped SMS messages from GPS tracker to GPX tracks
  Created: 26.02.2016
"""
import re
from datetime import datetime as datetime
import xml.etree.ElementTree as ET
import gpxpy.gpx as GPX

#Parse valid messages:
# lat:54.719937 long:20.525533 speed:0.00 dir:0.0\nT:21/02/16 01:45\nhttp://maps.google.com/maps?f=q&amp;q=54.719937,20.525533&amp;z=16

def parse_smses(cfg):
    str_time= '(?P&lt;Time&gt;\d\d/\d\d/\d\d \d\d\:\d\d)' #search time string '24/02/16 20:40'
    bWriteComments=         'b_write_comments' in cfg['process'] and cfg['process']['b_write_comments']
    bWriteWithoutCoord= 'b_write_without_coord' in cfg['process'] and cfg['process']['b_write_without_coord']
    bAllTimeFromStamp= 'b_all_time_from_stamp' in cfg['process'] and cfg['process']['b_all_time_from_stamp']
    #min_str_date= cfg['process']['min_str_date'] if 'min_str_date' in cfg['process'] else datetime.strptime('01.01.2010', '%d.%m.%Y')
    min_date= cfg['process']['min_date'] if 'min_date' in cfg['process'] else datetime.strptime('01.01.2010', '%d.%m.%Y')
    re_msgBad= re.compile('^(?P&lt;Comment&gt;[^T]*)(?:\nT\:)' + str_time + '.*')

    re_msg= re.compile('^(?P&lt;Comment&gt;[^\:]*?)(?: ?lat[:])(?P&lt;Lat&gt;\d{1,3}\.\d{1,8}) (?:long[:])(?P&lt;Lon&gt;\d{1,3}\.\d{1,8}) \
(?:speed[:])(?P&lt;Speed&gt;\d{1,3}\.\d{1,3}) (?:dir[:])(?P&lt;Dir&gt;\d{1,3}).*(?:\nT[:])' + str_time)
    kmh2ms= 1000.0/3600.0

    getFromStamp= lambda n: datetime.utcfromtimestamp(long(n.attrib['date'] if \
len(n.attrib['date_sent'])&lt;=1 else n.attrib['date_sent'])/1000.0) #'date2'
    def getTime(node= None, m_groupTime= None):
        if bAllTimeFromStamp:
            time_b= getFromStamp(node)
            if time_b &lt; min_date:
                return None
        else:
            try:
                time_b= datetime.strptime(m_groupTime, r'%d/%m/%y %H:%M') - cfg['input_files']['dt_from_utc']
                if time_b &lt; min_date: raise ValueError
            except ValueError:
                time_b= getFromStamp(node)
                if time_b &gt; min_date:
                    print('bad time: "' + m_groupTime +'" replaced by stamp')
                else:
                    return None
        return time_b

    tree = ET.parse(cfg['input_files']['path'])
    root = tree.getroot()

    # Create tracks in our GPX
    gpx = GPX.GPX()
    gpx_track= {}
    #gpx.tracks= [GPX.GPXTrack(name = c) for c in contact_names]
    #gpx_track= dict(zip(contact_names, gpx.tracks))
    for contact_name in cfg['input_files']['contact_names']:
        contact_name_d= contact_name.decode('cp1251') #allow Russian letters in contact names
        try:
            print('contact name: ', contact_name_d)
        except UnicodeEncodeError:
            contact_name_d= contact_name.decode('utf-8') #.encode('cp1251')
        time_b_prev= datetime.now() #or any big value
        gpx_track[contact_name]= GPX.GPXTrack(name = contact_name_d)
        #if gpx_track[contact_name].number: #.get_points_no():
        print('contact name: ', contact_name_d)
        #else:
            #continue
        gpx.tracks.append(gpx_track[contact_name])
        # Create segment in this GPX track:
        gpx_segment = GPX.GPXTrackSegment()
        gpx_track[contact_name].segments.append(gpx_segment)
        for neighbor in root.findall("./sms/[@contact_name='"+ contact_name_d +"'][@type='1']"): #received SMS
            #Parse messages:
            m= re_msg.search(neighbor.attrib['body']) #get ('54.738238', '20.503257', '0.00', '0', '24/02/16 20:40')
            if (m is not None) and (m.group(1) is not None): # valid =&gt; create points:
                Comment= m.group('Comment') if bWriteComments else None
                time_b= getTime(neighbor, m.group('Time'))
                if not time_b:
                    continue
                # Add segment for big time intervals:
                if time_b - time_b_prev &gt; cfg['process']['dt_between_track_segments']:
                    gpx_segment = GPX.GPXTrackSegment()
                    gpx_track[contact_name].segments.append(gpx_segment)
                time_b_prev= time_b
                if m.group('Speed')&lt;&gt;'0.00':
                    speed_b= round(float(m.group('Speed'))*kmh2ms, 5)
                else:
                    speed_b= '0' # work around: gpxpy not writes 0 if float
                gpx_point= GPX.GPXTrackPoint(latitude= m.group('Lat'), longitude= m.group('Lon'),
                                             time= time_b, speed= speed_b, comment= Comment)
                gpx_point.course= m.group('Dir') # where to write dir?
            elif bWriteWithoutCoord:                        # invalid =&gt; messages
                m= re_msgBad.search(neighbor.attrib['body'])
                if (m is not None) and (m.group(1) is not None): # valid time=&gt; create comment
                    Comment= m.group('Comment') if bWriteComments else None
                    time_b= getTime(neighbor, m.group('Time'))
                    if not time_b:
                        continue
                    gpx_point= GPX.GPXTrackPoint(time= time_b, comment= Comment)
                else: #get Time from message receive time stamp
                    Comment = neighbor.attrib['body'] if bWriteComments else None #Consider full message as comment
                    try:
                        time_b= getFromStamp(neighbor)
                        if time_b &lt; min_date:
                            continue #raise ValueError
                    except:
                        time_b= None
                        print('can not find time in message and time stamp')
                        continue
                    gpx_point= GPX.GPXTrackPoint(time= time_b)
            else:
                continue
            gpx_segment.points.append(gpx_point)
        gpx.description= contact_name_d
        gpx.author_email= 'andrey.korzh@atlantic.ocean.ru'
    return gpx

if __name__ == '__main__':
    if __package__ is None:
        from sys import path as sys_path
        from os  import path as  os_path
        sys_path.append(os_path.dirname(os_path.dirname(os_path.abspath(__file__))))
    from callFunc import ini2dict, init_logging, Ex_nothing_done, prep
    import logging
    import argparse

    cfg= ini2dict(r'SMS2GPX.ini') #d:\Work\_Python\_fromMat\

    parser = argparse.ArgumentParser( description= 'Convert SMS *.xml to *.gpx',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    epilog= 'If use special characters then insert arguments in quotes',
    version= r'''SMS2GPX version 0.0.1 - (c) 2016 Andrey Korzh &lt;korzh@nextmail.ru&gt;.'''
)
    parser.add_argument('path', nargs='?', type=str, default='.',
                        help='Path to XML or directory with XML files to parse')
    parser.add_argument('out', nargs='?', type=str, default='./&lt;filename&gt;.gpx',
                        help='''Output dir/path.
Join data from all found input files to single output if extension provided. If
"&lt;filename&gt;" found it will be sabstituted with [1st file name]+, if "&lt;dir&gt;" -
with last directory name.
Else, if no extension provided then ".gpx" will be used, "&lt;filename&gt;" string
will be sabstituted with correspondng input file names.
''')
    parser.add_argument('-verbose', nargs= 1, type=str, default= ['INFO'],
        choices= ['CRITICAL','ERROR','WARNING','INFO','DEBUG','NOTSET'],
        help= 'Verbosity of messages in log file')

    args = parser.parse_args()
    args.verbose= args.verbose[0]
    if args.path=='.': #len(sys.argv)==1:
        args.path= cfg['input_files']['path']
    if not 'dt_from_utc' in cfg['input_files']:
        cfg['input_files']['dt_from_utc']= 0


    # Input files
    default_input_filemask= '*.xml'
    inD, namesFE, nFiles, outD, outF, outE, bWrite2dir, msgFile= prep(args, default_input_filemask)
    #inD, inMask, inDlast= splitPath(args.path, default_input_filemask)

    l= init_logging(logging, outD, None, args.verbose)
    l.warn(msgFile)


    gpx= parse_smses(cfg)
    try:
        f= open(cfg['input_files']['path'][:-4] + '.gpx', 'w')
        bMissedCoordTo0= 'b_missed_coord_to_zeros' in cfg['process'] and cfg['process']['b_missed_coord_to_zeros']
        if bMissedCoordTo0:
            for p in gpx.walk(only_points=True):
                if p.latitude is None or p.longitude is None:
                    p.latitude = '0' #float('NaN') #0
                    p.longitude= '0' #float('NaN') #0
                #if p_prev==p:
                    #p.delete
                #p_prev= p


        #gpx.add_missing_data() #remove_empty()
        f.write(gpx.to_xml())
        print('ok')
    except Ex_nothing_done as e:
        print(e.message)
    except Exception as e:
        l.error((e.msg if hasattr(e,'msg') else '') + msg_option)
    finally:
        f.close()
        try:
            if not bWrite2dir:
                fp_out.close()
            #l.handlers[0].flush()
            logging.shutdown()
        except:
            pass


'''
TRASH

#to reformat time string:
#'24/02/16 20:40' to GPX.DATE_FORMAT='%Y-%m-%dT%H:%M:%SZ'
re_msgBad= re.compile('(\d\d)/(\d\d)/(\d\d) (\d\d\:\d\d)') #useful fields in parenthesies
bodyTimeStr, ok= re_msgBad.subn(r'\3-\2-\1T\4:00Z', m.group('Time'), count=1)

datetime.fromtimestamp(time_b/1000, pytz.timezone('UTC'))
'''
</t>
</tnodes>
</leo_file>
