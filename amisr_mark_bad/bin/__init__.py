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
    parser.add_argument('--beam', type=int, default=0, help='Beam index to plot first.')
    parser.add_argument('--maxbeams', type=int, default=0, help='Maximum number of beams to load.')
    parser.add_argument('--param_group', action='store', type=str, default='FittedParams', help='Group of the parameter to plot.')
    parser.add_argument('--param2plot', action='store', type=str, default='Ne', help='Group of the parameter to plot.')
    parsed_args = parser.parse_args(args)

    port = parsed_args.port
    param_group = parsed_args.param_group
    param2plot = parsed_args.param2plot
    beam = parsed_args.beam
    maxbeams = parsed_args.maxbeams
    appfile = os.path.join(amisr_mark_bad.__path__[0],"mark_data.py")
    portcmd = f"--port {port}"
    argscmd = f"--args --param_group {param_group} --param2plot {param2plot} "\
            f"--beam {beam} --maxbeams {maxbeams} {parsed_args.fitter_file}"
    command_line = f"bokeh serve {portcmd} {appfile} {argscmd}"
    print(f"cmd:{command_line}")
    try:
        process = subprocess.Popen( shlex.split(command_line),bufsize=1)
        process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        process.wait()
    #def signal_handler(signal, frame):
    #    print('You pressed Ctrl+C!')
    #    sys.exit(0)

    #signal.signal(signal.SIGINT, signal_handler)
    #print('Press Ctrl+C')

    #signal.pause()
if __name__ == "__main__":
    main()
