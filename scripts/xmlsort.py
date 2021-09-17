#!python3
"""
Sorts second level tags and attributes in xml file/s.
If argv[1] is glob then for each file writing to file with name of input modified by adding argv[2] ("_xmlsrt" if 1 arg)
 and same extension
skips to output that exist
Was used to unify ODV *.xview files before compare
"""

#from lxml import etree
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from xml.etree.ElementTree import parse, ParseError
import sys
n_args = len(sys.argv)-1
print(n_args, 'arguments:', sys.argv[1:])
if n_args < 1:
    print('usage:\nxmlsort.py file_in.xml file_out.xml\n'
          'or\nxmlsort.py filemask name_suffix\n'
          'File_out or filemask must contain suffix and name_suffix must not contain dot (and so no suffixes)')
    exit(0)

filename_in = Path(sys.argv[1])
if n_args > 1 and ('.' in sys.argv[2]):
    out_type = 'file'
    filename_in = [filename_in]
    filename_out_fun = lambda file_in: Path(sys.argv[2])
else:
    out_type = 'files'
    sfx = filename_in.suffix
    name_sfx = sys.argv[2] if n_args > 1 else '_xmlsrt'
    filename_in = Path(os.getcwd()).glob(str(filename_in))
    filename_out_fun = lambda file_in: file_in.with_name(f'{file_in.stem}{name_sfx}').with_suffix(sfx)


def standard_error_info(e):
    msg_trace = '\n==> '.join((s for s in e.args if isinstance(s, str)))
    return f'{e.__class__}: {msg_trace}'


def keys_to_sort(elem):
    return elem.tag, elem.get('desc')


def sort_attrib(elem):
    if elem.attrib:
        elem.attrib = dict(sorted(elem.attrib.items()))


print('Processing', filename_in, out_type, '...')
file_out = ''
for file_in in filename_in:
    if file_in == file_out:
        continue
    file_out = filename_out_fun(file_in)
    if file_out.is_file():
        print(file_in.name, f'- {file_out.name} exist: skipped')
        continue
    print(file_in.name, '-', file_out.name)
    try:
        tree = parse(file_in)  # fromstring(xmlstr)
    except ParseError as e:
        print(standard_error_info(e), ' - skip')
        continue

    root = tree.getroot()

    # skip sort the first layer
    # root[:] = sorted(root.iter(), key=lambda child: (child.tag,child.get('name')))

    # sort the second layer
    for c in root.iter():
        c[:] = sorted(c, key=keys_to_sort)
        sort_attrib(c)


    # def getSortValue(elem):
    #     if isinstance(elem,etree._Comment):
    #         # sort comment by its content
    #         return elem.text
    #     else:
    #         # sort entities by tag and then by name
    #         return elem.tag + elem.attrib.get('name','')
    #
    # doc=etree.parse(filename_in)
    #
    # for parent in doc.xpath('//*[./*]'): # Search for parent elements
    #     parent[:] = sorted(parent, key=lambda x: getSortValue(x))
    # (etree.tostring(doc, pretty_print=True))

    tree.write(file_out, xml_declaration=True, encoding='utf-8')
