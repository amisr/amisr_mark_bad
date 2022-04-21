"""
File name : __init__.py

description : interface to the command line parsing of arguments

Author: Pablo M. Reyes
History:
    20 Arp 2022: First implementation
"""
import os
import sys
import shlex
import signal
import subprocess
import argparse
import amisr_mark_bad

def main(args=None):
    parser=argparse.ArgumentParser(
        description='''AMISR processed file bad block identification.''')
    parser.add_argument('fitter_file', help='Fitted file.')
    parser.add_argument('--port', type=int, default=5007, \
            help='serving port. Default=5007')
    parsed_args = parser.parse_args(args)

    appfile = os.path.join(amisr_mark_bad.__path__[0],"mark_data.py")
    portcmd = f"--port {parsed_args.port}"
    argscmd = f"--args {parsed_args.fitter_file}"
    command_line = f"bokeh serve {portcmd} {appfile} {argscmd}"
    print(f"cmd:{command_line}")
    process = subprocess.Popen( shlex.split(command_line),bufsize=1)

    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C')

    signal.pause()
if __name__ == "__main__":
    main()
