import time
import datetime

#1st, 6th, 24th

def get_hours_time(set_hours):
    hours = datetime.datetime.now()-datetime.timedelta(hours=set_hours)
    unixtime = int(time.mktime(hours.timetuple()))
    return str(unixtime)

