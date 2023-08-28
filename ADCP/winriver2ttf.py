#!/usr/bin/env python
# coding:utf-8


"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>
  Purpose: Create .ttf fle (XML) for WinRiver II txt output using list of needed
  column names in .ini file and txt file with lookup list of numbers with
  corresponding names obtained from Winriver II pdf documentation
  Created: 26.05.2016
"""
import re
from lxml import etree as ET  # import lxml.etree
from pathlib import Path
rs = '(?P<val>^\d{1,3})\. *(?P<key>[^\n;]+)(?:.*$)'
ro = re.compile(rs, re.MULTILINE)  # |re.IGNORECASE


def cfg_prepare(cfg):
    '''Get list of numbers to insert in output'''
    columns = [(cfg['columns'][k.lower()] if k.lower() in cfg['columns'] else k) \
               for k in cfg['out']['colmn_names']]
    # get number slist
    ext = os_path.splitext(cfg['in']['path'])[1]
    out_numbers = []
    if ext == '.txt':
        # really bad
        with open(cfg['in']['path'], 'r') as fp_l:
            read_data = fp_l.read()
        lt = ro.findall(read_data)
        # m= dict(zip(*zip(*lt)[-1::-1]))
        # lkeys= zip(*lt)[1]
        for col in columns:
            for v, k in lt:
                if k.lower().startswith(col.lower()):
                    out_numbers.append(int(v) - 1)
                    break
            else:
                out_numbers.append(0)
    elif ext == '.xml':
        # root = ET.parse(cfg['in']['path'])
        # <NGSP>
        #    <Schema>
        #        <Member
        parent = ET.parse(cfg['in']['path']).getroot()[0]
        els = parent.getchildren()
        assert parent.tag == 'Schema'
        for col in columns:
            for v, el in enumerate(els):
                if el.text.lower().startswith(col.lower()):
                    out_numbers.append(v)
                    break
            else:
                out_numbers.append(0)
    return out_numbers


if __name__ == '__main__':
    # unittest.main()
    if __package__ is None:
        from sys import path as sys_path
        from os import path as  os_path

        sys_path.append(os_path.dirname(os_path.dirname(os_path.abspath(__file__))))
    from utils2init import ini2dict, pathAndMask, name_output_file  # dir_walker, readable, bGood_dir, bGood_file

    cfg = ini2dict()

    cfg['out']['digits'] = cfg_prepare(cfg)
    if 'path' not in cfg['out']:
        cfg['out']['path'] = cfg['in']['path'] / os_path.basename(cfg['out']['name'])
    cfg['out']['path'] = Path(*pathAndMask(*[cfg['out'][spec] if spec in cfg['out'] else None for \
        spec in ['path', 'name', 'ext']]))
    cfg['out']['path'], writeMode, msgFile = name_output_file(
        cfg['out']['path'].parent, cfg['out']['path'].stem, cfg['out']['path'].suffix, False)
    root = ET.Element('TTF')
    child1 = ET.SubElement(root, 'Subscription')
    child1.set('Id', "WinRiver Processed Data")
    child11 = ET.SubElement(ET.SubElement(child1, 'Translator'), 'XMLFile')
    ET.SubElement(child11, 'Name').text = "Proc_Ensemble_schema.xml"
    ET.SubElement(child11, 'Title').text = "Processed Ensemble Data"
    child2 = ET.SubElement(root, 'Settings')
    child21 = ET.SubElement(child2, 'TabularData')  # Parent for elements in cycle

    child_pattern = ET.Element('Selected')
    ET.SubElement(child_pattern, 'Translator').text = '0'
    child_pattern_digit = ET.SubElement(child_pattern, 'Data')
    from copy import copy

    for d in cfg['out']['digits']:
        child_pattern_digit.text = str(d)
        child21.append(copy(child_pattern))  # ET.Element()

    ET.SubElement(child21, 'Output').text = '2'
    ET.SubElement(child21, 'OutputFileName').text = 'a'  # "&lt;Файл не Выбран&gt;"
    ET.SubElement(child21, 'IncludeHeader').text = '0'
    ET.SubElement(child21, 'CustomParamDelimiter').text = '9'
    ET.SubElement(child21, 'MessageDelimiter').text = '2'
    ET.SubElement(child21, 'CustomMsgDelimiter').text = '0'
    # Write
    et = ET.ElementTree(root)
    if msgFile:
        msgFile = '(' + msgFile + ')'
    try:
        et.write(cfg['out']['path'], pretty_print=True)
        print('Saved to', cfg['out']['path'], msgFile)
    except IOError:
        print('Can not save to', cfg['out']['path'], msgFile)
        et.write(cfg['out']['path'], pretty_print=True)
