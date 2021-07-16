from pathlib import Path
import logging
from typing import Any, Callable, Dict, Iterator, Mapping, MutableMapping, Optional, List, Sequence, Tuple, Union

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import base64
# import email
# from datetime import datetime
# from bs4 import BeautifulSoup

from utils2init import LoggingStyleAdapter
lf = LoggingStyleAdapter(logging.getLogger(__name__))

# Define the SCOPES. If modifying it, delete the token.pickle file.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def get_gmails_data(json_cred, q, parse_body: Callable[[str], Any], cfg_dir=None):
    """

    :param json_cred: json file name downloaded from https://console.cloud.google.com/apis/credentials
    :param q:
    :param parse_body:
    :param cfg_dir: to read checked token (save to *.picle 1st time). If None then same path and stem as json_cred used
    :return:
    """
    print('reading messages from my email...')
    data = []

    # Variable creds will store the user access token.
    # If no valid token found, we will create one.
    creds = None

    # The file token.pickle contains the user access token.
    # Check if it exists
    file_with_token = (Path(cfg_dir) / 'google_token.pickle') if cfg_dir else Path(json_cred).with_suffix('.pickle')
    if file_with_token.exists():
        # Read the token from the file and store it in the variable creds
        with file_with_token.open('rb') as token:
            creds = pickle.load(token)

    # If credentials are not available or are invalid, ask the user to log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(json_cred, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the access token in token.pickle file for the next run
        with file_with_token.open('wb') as token:
            pickle.dump(creds, token)

    # Connect to the Gmail API
    service = build('gmail', 'v1', credentials=creds)

    # request a list of all the messages
    result = service.users().messages().list(userId='me', q=q, maxResults=1000000).execute()  # 'from: '

    # We can also pass maxResults to get any number of emails. Like this:
    # result = service.users().messages().list(maxResults=200, userId='me').execute()
    messages = result.get('messages')

    # messages is a list of dictionaries where each dictionary contains a message id.

    # iterate through all the messages
    for msg in messages:
        # Get the message from its id
        txt = service.users().messages().get(userId='me', id=msg['id']).execute()
        try:
            # Get value of 'payload' from dictionary 'txt'
            payload = txt['payload']
            if payload is None:
                continue
            headers = payload['headers']

            # Look for Subject and Sender Email in the headers
            for d in headers:
                if d is None:
                    continue
                if d['name'] == 'Subject':
                    subject = d['value']
                if d['name'] == 'From':
                    sender = d['value']

            body = base64.b64decode(payload['body']['data'].replace('-', '+').replace('_', '/'))

            # # Get the data and decode it with base 64 decoder.
            # parts = payload.get('parts')[0]
            # data = base64.b64decode(parts['body']['data'].replace('-', '+').replace('_', '/'))

            # # Now, the data obtained is in lxml. So, we will parse
            # # it with BeautifulSoup library
            # soup = BeautifulSoup(data , "lxml")
            # body = soup.body()

            # # Printing the subject, sender's email and message
            # print("Subject: ", subject)
            # print("From: ", sender)
            # print("Message: ", body)
            # print('\n')
            data.append(parse_body(body.decode(errors='ignore')))

        except Exception as e:
            lf.exception('Read message')
            pass
    return data


#get_gmails()
