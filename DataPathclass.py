# import os
import platform


class DataPath(object):
    def __init__(self):
        if platform.system()=='Darwin':   # on mac
            self.sysPathHeader = '/Volumes/TOSHIBA/'
        else:   # on linux
            self.sysPathHeader = '/media/'

