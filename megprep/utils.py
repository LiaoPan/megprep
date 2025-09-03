#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import yaml
import re
import os
import time
import getpass
import subprocess
import random
import socket
import redis
import numpy as np
import argparse

def handle_yaml_scientific_notation():
    # handle scientific notation.
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    return loader


def str2bool(value):
    if value.lower() in ('true', 't', '1'):
        return True
    elif value.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_random_seed(seed=None):
    """Set all random seeds.

    This includes Python's random module and NumPy.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)

    print(f"Setting random seed to {seed}")

    random.seed(seed)
    np.random.seed(seed)
    return seed



# def start_xvfb():
#     """Xvfb restart."""
#     try:
#         print("Try to restart Xvfb service...")
#         os.system('ps aux|grep super')
#         os.system('ps aux|grep Xvfb')
#         print("supervisord restarted...")
#
#         os.system('/usr/bin/supervisord  -c /etc/supervisor/conf.d/supervisord.conf')
#         os.system('ps aux|grep super')
#         os.system('ps aux|grep Xvfb')
#         # os.system('Xvfb :99 -screen 0 1920x1080x24 &')
#         os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "150"
#         os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.2"
#         os.environ['DISPLAY'] = ':99'
#         os.environ["QT_QPA_PLATFORM"] = "xcb"
#
#         time.sleep(3)
#         print("Double Check.....")
#         os.system('ps aux|grep super')
#         os.system('ps aux|grep Xvfb')
#         return True
#     except Exception as e:
#         print(f"Notice: Restart Xvfb failed!{e}")
#         return False


def get_xvfb_processes():
    """Retrieve a list of all running Xvfb processes along with their owners."""
    xvfb_processes = []
    try:
        # Use ps command to get information about all Xvfb processes
        result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE, text=True)
        for line in result.stdout.splitlines():
            if 'Xvfb' in line and 'grep' not in line:
                parts = line.split()
                pid = parts[1]  # Process ID
                user = parts[0]  # Owner of the process
                xvfb_processes.append((pid, user))
    except Exception as e:
        print(f"Unable to retrieve process information: {e}")

    return xvfb_processes


def kill_non_owning_processes(xvfb_processes, current_user):
    """Terminate Xvfb processes not owned by the current user."""
    for pid, user in xvfb_processes:
        if user != current_user:
            try:
                print(f"Killing process: PID={pid}, User={user}")
                subprocess.run(['kill', pid])
            except Exception as e:
                print(f"Unable to kill process {pid}: {e}")


def is_xvfb_running():
    try:
        current_user = getpass.getuser()

        result = subprocess.run(
            ['ps', '-u', current_user, '-o', 'pid,cmd'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        xvfb_running = False
        output_lines = result.stdout.strip().split('\n')[1:]

        for line in output_lines:
            if 'Xvfb' in line:
                print(f"Xvfb is running for the current user:{current_user}")
                print(line.strip())
                xvfb_running = True

        return xvfb_running

    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e.stderr.strip()}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def find_free_display():
    """Find a free display number by checking port numbers."""
    while True:
        # Generate a random display number between 1000 and 65535
        display_number = random.randint(1000, 65535)
        # Check if the port is free
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(('localhost', display_number)) != 0:
                return display_number


def start_xvfb():
    """Start Xvfb on a randomly chosen free display and set the DISPLAY environment variable."""
    # Find a free display number
    free_display = find_free_display()
    display_str = f":{free_display}"  # Create display string
    print(f"Starting Xvfb on {display_str}...")

    # Start Xvfb
    # os.system(f'Xvfb {display_str} -screen 0 1920x1080x24 &')
    os.system(f'Xvfb {display_str} -screen 0 1024x768x24 &') # low
    # os.system('/usr/bin/supervisord  -c /etc/supervisor/conf.d/supervisord.conf')
    os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "150"
    os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.2"
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    # Set the DISPLAY environment variable
    os.environ['DISPLAY'] = display_str
    print(f"DISPLAY environment variable set to {os.environ['DISPLAY']}")

    return free_display


def stop_xvfb(display_number):
    """Stop the Xvfb process associated with the specified display number."""
    try:
        display_str = f":{display_number}"
        # Use pgrep to find the PID of running Xvfb
        pid = subprocess.check_output(["pgrep", "-f", f"Xvfb {display_str}"]).strip()
        pid_str = pid.decode("utf-8")
        if pid:
            # Kill the Xvfb process
            os.system(f'kill {pid_str}')
            # print(f"Xvfb on {display_str} stopped.")
        else:
            print(f"No Xvfb process found for {display_str}.")
    except subprocess.CalledProcessError:
        print(f"No Xvfb process found for {display_str}.")

# display_number = start_xvfb()
# import time
# time.sleep(10)
# stop_xvfb(display_number)
# xvfb_processes = get_xvfb_processes()
# current_user = getpass.getuser()
# kill_non_owning_processes(xvfb_processes, current_user)

class RedisGlobalVariableManager:
    def __init__(self, host='localhost', port=6379, db=0):
        """Initialize the Redis connection."""
        self.r = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)

    def set_global_variable(self, key, value):
        """Set a global variable in Redis."""
        self.r.set(key, value)

    def get_global_variable(self, key):
        """Get a global variable from Redis."""
        value = self.r.get(key)
        return value