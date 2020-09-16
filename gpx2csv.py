#!/usr/bin/env python
# coding:utf-8
import csv
import logging

import lxml.etree

from utils2init import init_file_names, Ex_nothing_done, cfg_from_args, my_argparser_common_part, \
    this_prog_basename, init_logging

''' From http://stackoverflow.com/questions/20714038/xml-to-csv-in-python:

This is a namespaced XML document. Therefore you need to address the nodes using their respective namespaces.
The namespaces used in the document are defined at the top:

xmlns:tc2="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xmlns:tp1="http://www.garmin.com/xmlschemas/TrackPointExtension/v1"
xmlns="http://www.topografix.com/GPX/1/1"

So the first namespace is mapped to the short form tc2, and would be used in an element like <tc2:foobar/>.
The last one, which doesn't have a short form after the xmlns, is called the default namespace, and it applies
to all elements in the document that don't explicitely use a namespace - so it applies to your <trkpt /> elements as well.

Therefore you would need to write root.iter('{http://www.topografix.com/GPX/1/1}trkpt') to select these elements.

In order to also get time and elevation, you can use trkpt.find() to access these elements below the trkpt node,
and then element.text to retrieve those elements' text content (as opposed to attributes like lat and lon).
Also, because the time and ele elements also use the default namespace you'll have to use the {namespace}
element syntax again to select those nodes.

So you could use something like this:
'''


# NS = 'http://www.topografix.com/GPX/1/0'
def gpx2csv(gpx_file_name):
    NS = 'http://www.topografix.com/GPX/1/1'
    header = ('time', 'lat', 'lon', 'cmt')
    header_b = [h.encode('ascii') for h in header]
    with open(gpx_file_name[:-4] + '.csv', 'w', newline='') as f:  # 'wb',
        writer = csv.writer(f, delimiter='\t')  # ,delimiter=' '
        writer.writerow(header_b)
        root = lxml.etree.parse(gpx_file_name)
        for trkpt in root.iter('{%s}trkpt' % NS):  # '{%s}trkpt' % NS
            lat = trkpt.get('lat')
            lon = trkpt.get('lon')
            time = trkpt.find('{%s}time' % NS).text  #
            node_cmt = trkpt.find('{%s}cmt' % NS)  #
            cmt = node_cmt.text.replace(',', ' ') if not node_cmt is None else ""
            writer.writerow((time, lat, lon, cmt))


def my_argparser():
    """
    Configuration parser
    - add here common options for different inputs
    - add help strings for them
    :return p: configargparse object of parameters
    """

    p = my_argparser_common_part({'description': """               
----------------------------
Convert data from GPX files
to CSV
----------------------------"""})

    # Configuration sections

    # All argumets of type str (default for add_argument...), because of
    # custom postprocessing based of args names in ini2dict
    p_in = p.add_argument_group('in', 'all about input files')
    p_in.add('--path', default='.',  # nargs=?,
             help='path to source file(s) to parse. Use patterns in Unix shell style')


def main(new_arg):
    cfg = cfg_from_args(my_argparser(), new_arg)
    if not cfg:
        return
    if cfg['program']['return'] == '<cfg_from_args>':  # to help testing
        return cfg

    l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
    print('\n' + this_prog_basename(__file__), end=' started. ')
    try:
        cfg['in'] = init_file_names(cfg['in'], cfg['program']['b_interact'])
    except Ex_nothing_done as e:
        print(e.message)
        return ()

    # cfg = {'in': {}}
    # cfg['in']['path'] = \
    #     r'd:\workData\BalticSea\181005_ABP44\navigation\2018-10-06tracks_copy.gpx'
    # r'd:\WorkData\_experiment\_2017\tracker\170502.gpx'
    # r'd:\workData\_experiment\2016\GPS_tracker\sms_backup\sms-20160225135922.gpx'
    for ifile, nameFull in enumerate(cfg['in']['paths'], start=1):
        print('{}. {}'.format(ifile, nameFull), end=', ')
        gpx2csv(nameFull)


if __name__ == '__main__':
    main()
