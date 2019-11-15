import socket 

from datetime import datetime


def TaskID_Generator():
    currentTime = datetime.now()
    strTime = "%04d%02d%02d_%02d%02d%02d" %(currentTime.year, currentTime.month, currentTime.day,currentTime.hour, currentTime.minute, currentTime.second)
    return strTime