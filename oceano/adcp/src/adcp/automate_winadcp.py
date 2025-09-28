"""
Try to automate WinADCP export. Not finished.
"""
import pyautogui as a
import subprocess
import sys
import time
from pathlib import Path
import logging

# if __name__ != '__main__':
l = logging.getLogger(__name__)

# Mask of input files
in_path = Path(r'D:\WorkData\KaraSea\231110_AMK93\ADCP_75kHz\_raw\AMK93[0-9][0-9]*.STA')

in_files = list(in_path.parent.glob(in_path.name))
if len(in_files) == 0:
    l.error('no %s files', in_path.name)
    exit()
else:
    print(len(in_files), 'files found, converting...')

# # start the app in a separate process using the same interpreter as this script
# process = subprocess.Popen([sys.executable, 'our_app.py'])

def wait_for_windows(titles):
    for i in range(200):
        for title in titles:
            windows = a.getWindowsWithTitle(title)
            if len(windows) >= 1:
                props = {iw: w.box for iw, w in enumerate(windows) if w.title == title}
                if len(props) > 1:
                    l.warning('windows with different properties found for %s: %s', title, ','.join(props.values()))
                return windows[props[0]]
        time.sleep(0.1)
    else:
        l.error('expected %s windows not found', ' or '.join(titles))
        return


def activate(win):
    win.restore()
    a.move(*win.center)
    a.click(button='left')


winadcp = r"C:\Program Files (x86)\RD Instruments\WinADCP\WinADCP.exe"
max_rep = 3

# Open the .exe program
with subprocess.Popen(winadcp) as process:
    for ifile, file in enumerate(in_files):
        b_1st_try = True
        for rep in range(max_rep+1):

            # restart if needed
            if not b_1st_try:
                print(f'Restarting WinADCP ({rep}/{max_rep})...')
                process.terminate()
                process = subprocess.Popen(winadcp)
            b_1st_try = False

            # wait for the window
            window = wait_for_windows(['WinADCP'])
            if not window:
                continue  # try again

            # Set foreground focus to the window
            while a.getActiveWindow() != window:  # .isActive hangs!!!!!!!!!
                # a.hotkey("alt","tab")
                print('set active WinADCP window')
                window.activate()
                time.sleep(1)               # window[-1].set_foreground()

            a.hotkey('alt', 'f', 'o')  # Open file dialog
            wo = wait_for_windows(['Открытие', 'Open'])
            if not wo:
                if a.getActiveWindow() != window:
                    print('set active!')
                    window.activate()
                    time.sleep(1)
                # window.minimize()
                # window.restore()
                a.hotkey('alt', 'f', 'o')
                wo = wait_for_windows(['Открытие', 'Open'])
                if wo:
                    continue  # try again
            wo.activate()
            a.write(f'{file}')  # enter file name
            a.press('enter')

            # Wait loading data
            # why wrong?:
            # window = wait_for_windows(['WinADCP'])

            time.sleep(5)  # only option worked

            while a.getActiveWindow() != window:
                print('set active after loading data')
                window.activate()
                time.sleep(1)
            a.hotkey('alt', 'x')  # Export
            wo = wait_for_windows(['WinADCP Export Options'])
            while not wo:
                while a.getActiveWindow() != window:
                    print('cannot open export options!')
                    a.hotkey("alt","tab")
                    window.activate()
                    time.sleep(1)
                a.hotkey('alt', 'x')  # Export
                wo = wait_for_windows(['WinADCP Export Options'])
                time.sleep(1)

            # wo.activate()
            a.hotkey('ctrl', 'tab')  # Series/Ancsilary
            a.hotkey('ctrl', 'tab')

            for i in range(6):

                a.hotkey('shift', 'tab')  # Lat/Lon
            a.press('space')

            for i in range(4):  # Bins All
                a.hotkey('tab')
            a.press('space')

            if a.getActiveWindow() != wo:
                print('set active export options')
                wo.activate()
            for i in range(5):  # mat format
                a.hotkey('tab')
            a.press('down')

            a.hotkey('shift', 'tab')  # Write file button
            a.hotkey('shift', 'tab')
            a.press('enter')
            if not wait_for_windows(['Сохранение', 'Save']):
                continue  # try again
            out_file = file.with_name(file.stem + '_STA').with_suffix('.mat')
            if out_file.is_file():
                out_file.unlink()  # remove old file as WinADCP can't overwrite it
                print('Old file removed')

            a.write(f'{out_file.name}')  # Enter file name
            a.press('enter')
            time.sleep(1)

            a.hotkey('alt', 'F4')    # Close export options
            time.sleep(1)
            break  # next fileD:\WorkData\KaraSea\231110_AMK93\ADCP_75kHz\_raw\AMK93009_000000.STA


        print(ifile + 1, out_file.name, 'saved')


    # Close the program
process.terminate()