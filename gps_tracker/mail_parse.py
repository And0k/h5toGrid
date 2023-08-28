import sys
from typing import Any, Callable, Dict, Iterator, Mapping, MutableMapping, Optional, List, Sequence, Tuple, Union
from pathlib import Path
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import mailbox
from time import sleep
import pandas as pd

import logging
import imaplib

try:
    # from ruamel_yaml import safe_load
    from ruamel.yaml import safe_load
    # from ruamel.yaml import YAML
    # yaml = YAML()
    # yaml.indent(mapping=2, sequence=4, offset=2)
    # # yaml.explicit_start=False
except ImportError:
    from yaml import safe_load

from gps_tracker.gmail_read import get_gmails_data

# def read_email_from_gmail(smtp_server, from_email, from_pwd, **kwargs):
#     try:
#         mail = imaplib.IMAP4_SSL(smtp_server)
#         mail.login(from_email, from_pwd)
#         mail.select('inbox')
#
#         data = mail.search(None, 'ALL')
#         mail_ids = data[1]
#         id_list = mail_ids[0].split()
#         first_email_id = int(id_list[0])
#         latest_email_id = int(id_list[-1])
#
#         for i in range(latest_email_id, first_email_id, -1):
#             data = mail.fetch(str(i), '(RFC822)')
#             for response_part in data:
#                 arr = response_part[0]
#                 if isinstance(arr, tuple):
#                     msg = email.message_from_string(str(arr[1], 'utf-8'))
#                     email_subject = msg['subject']
#                     email_from = msg['from']
#                     print('From : ' + email_from + '\n')
#                     print('Subject : ' + email_subject + '\n')
#
#     except Exception as e:
#         print(str(e))

def spot_from_gmail(device_number: Union[str, int], time_start: datetime):
    """

    :param device_number:
    :param time_start: time in utc zone
    :return:
    """
    ep = datetime(1970, 1, 1, tzinfo=timezone.utc)  # 'US/Pacific' PST -1 day, 16:00:00 STD

    #ep = pd.Timestamp('1969-12-31 19:00:00', tz='utc')  # datetime( 0, tzinfo=timezone.utc)
    return get_gmails_data(
        str(Path.home() / 'client_secret_2_310771020112-fbq4dukacte2nevs4d7kc5decga1cahb.apps.googleusercontent.com.json'
        # 'client_secret_310771020112-fbq4dukacte2nevs4d7kc5decga1cahb.apps.googleusercontent.com.json'
        ),
        q=f'subject:"Position Alert Activated: {device_number}" after:{round((time_start - ep).total_seconds())} from:alerts@maps.findmespot.com',
        parse_body=parse_spot_text
        )
    # cfg_mail = safe_load(f_yml)
    # txt = read_email_from_gmail(**cfg_mail)  # ['smtp_server'], cfg_mail['from_email'], ['from_pwd']



if False:

    import email
    from email.policy import default

    # Read a big .mbox file with Python
    # quick and dirty attempt to implement a generator to read in an mbox file message by message
    class MboxReader:
        def __init__(self, filename):
            self.handle = open(filename, 'rb')
            assert self.handle.readline().startswith(b'From ')

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
            self.handle.close()

        def __iter__(self):
            return iter(self.__next__())

        def __next__(self):
            lines = []
            while True:
                line = self.handle.readline()
                if line == b'' or line.startswith(b'From '):
                    yield email.message_from_bytes(b''.join(lines), policy=default)
                    if line == b'':
                        break
                    lines = []
                    continue
                lines.append(line)


    # Usage:
    with MboxReader(mboxfilename) as mbox:
        for message in mbox:
            print(message.as_string())
    ##################################################################################################


    def getcharsets(msg):
        charsets = set({})
        for c in msg.get_charsets():
            if c is not None:
                charsets.update([c])
        return charsets

    def handleerror(errmsg, emailmsg,cs):
        print()
        print(errmsg)
        print("This error occurred while decoding with ",cs," charset.")
        print("These charsets were found in the one email.",getcharsets(emailmsg))
        print("This is the subject:",emailmsg['subject'])
        print("This is the sender:",emailmsg['From'])

    def getbodyfromemail(msg):
        body = None
        #Walk through the parts of the email to find the text body.
        if msg.is_multipart():
            for part in msg.walk():

                # If part is multipart, walk through the subparts.
                if part.is_multipart():

                    for subpart in part.walk():
                        if subpart.get_content_type() == 'text/plain':
                            # Get the subpart payload (i.e the message body)
                            body = subpart.get_payload(decode=True)
                            #charset = subpart.get_charset()

                # Part isn't multipart so get the email body
                elif part.get_content_type() == 'text/plain':
                    body = part.get_payload(decode=True)
                    #charset = part.get_charset()

        # If this isn't a multi-part message then get the payload (i.e the message body)
        elif msg.get_content_type() == 'text/plain':
            body = msg.get_payload(decode=True)

       # No checking done to match the charset with the correct part.
        for charset in getcharsets(msg):
            try:
                body = body.decode(charset)
            except UnicodeDecodeError:
                handleerror("UnicodeDecodeError: encountered.",msg,charset)
            except AttributeError:
                 handleerror("AttributeError: encountered" ,msg,charset)
        return body


mboxfile = (Path.home() / 'AppData' / 'Roaming' / 'Thunderbird' / 'Profiles' / 'qwd33vkh.default-release' / 'Mail' /
               'Local Folders' / 'AB_SIO_RAS_tracker') #.mozmsgs
# c:\Users\and0k\AppData\Roaming\Thunderbird\Profiles\qwd33vkh.default-release\Mail\Local Folders\AB_SIO_RAS_tracker
# c:\Users\and0k\AppData\Roaming\Thunderbird\Profiles\qwd33vkh.default-release\ImapMail\imap.gmail.com\[Gmail].sbd\Trash.mozmsgs\
# C:\Users\and0k\AppData\Roaming\Thunderbird\Profiles\qwd33vkh.default-release\Mail\Local Folders\AB_SIO_RAS_tracker\cur
    #'C:/Users/Username/Documents/Thunderbird/Data/profile/ImapMail/server.name/INBOX'


def parse_spot_text(body: str) -> Tuple[datetime, float, float]:
    """
    :param body:
    :return: Time, Lat, Lon
    """
    data = {}
    keys = ['Time', 'Lat/Lng']
    for key in keys:
        for i, row in enumerate(body.splitlines()):
            # More filtering
            # esn='3125300'
            # if not row[0].contains(esn):
            #     return None
            if row.startswith(key):
                try:
                    data[key] = row.split(' : ', maxsplit=1)[1].strip()
                    break
                except IndexError:  # list index out of range
                    continue        # -> try next row
    return [datetime.strptime(data['Time'], '%m/%d/%Y %I:%M:%S %p')] + [float(k) for k in data['Lat/Lng'].split(', ')]


def spot_tracker_data_from_mbox(mboxfile, subject_end, time_start) -> List[Tuple[datetime, float, float]]:
    print('reading', mboxfile)
    data = []

    mbox = mailbox.mbox(mboxfile, create=False)
    for i in range(60):  # "for" not need because the problem solving method with _unlock_file() is worked ok
        try:
            mbox.lock()  # try prevent other apps save empty messages to it on filed write (as I experienced with Thunderbird) - not helps
            break
        except mailbox.ExternalClashError:
            sleep(2)
            try:
                mailbox._unlock_file(mbox._file)
                continue
            except Exception as e:
                print('Can not unlock', e)
            print('waiting for locking')
    else:
        print('Can not lock mailbox')
        sys.exit(1)
    try:
        for key, thisemail in mbox.items():
            subject = thisemail['Subject']
            if not subject:
                print('Message #', key, 'have no subject: locked?')
            elif not subject.endswith(subject_end):
                continue
            try:
                receive_time = parsedate_to_datetime(thisemail['Date'])
                if receive_time < time_start:
                    continue
            except TypeError:
                try:
                    # don't know why I see date in _from byt it is last chance
                    receive_time = datetime.strptime(thisemail._from.split(maxsplit=2)[-1], '%b %d %H:%M:%S %Y')
                    print('found date, in not normal way: ', receive_time)
                except ValueError:
                    print('can not get date')
                continue
            if not subject:
                raise FileNotFoundError('message without subject')
            body = thisemail.get_payload(decode=True).decode(thisemail.get_charsets()[0])  # getbodyfromemail(thisemail)
            data.append(parse_spot_text(body))
        return data
    except FileNotFoundError:
        sys.exit(1)  # todo: wait and retry
    finally:
        mbox.unlock()

if True:  # if __name__ == '__main__':
    def test_load_gmail():
        spot_from_gmail(device_number=2, time_start=datetime.fromisoformat('2021-07-18T18:00+00:00'))

    def test_spot_tracker_data_from_mbox():
        spot_tracker_data_from_mbox(mboxfile,
                                    subject_end=': 4',
                                    time_start=datetime.fromisoformat('2021-06-04T15:49:07+00:00')
                                    )

        body = 'The Position alert for 4 (0-3125300) has been triggered: \r\n\r\nTime           : 6/4/2021 9:49:24 PM \r\nAsset          : 4 (0-3125300) \r\nLat/Lng        : 54.624180, 19.760560 \r\nAddress        : Kaliningrad, Russia \r\nAltitude       : -4.0 m \r\nSpeed          : 0.0 kph \r\nLink           : https://maps.findmespot.com/Track?showPositionId=6b5d34eb-6dc5-eb11-80fc-90b11c455a5d&from=2021-06-03+21%3a49%3a24Z&to=2021-06-05+21%3a49%3a24Z&showAssetId=474518 \r\n\r\nEvents\r\n - Unlimited Track \r\n\r\n--\r\nTo unsubscribe from this alert, please use this link:\nhttps://maps.findmespot.com/alerts/unsubscribe?id=17060&email=ao.korzh%40gmail.com&signature=%242a%2408%24CesN8ynWzYIHJDBuUczsw.hpN9teppzuYuH5DOLYxoq7%2FA3qde2QG\n\n\n'
