import time
import datetime

def get_24_hours_time():
    last_24 = datetime.datetime.now()-datetime.timedelta(hours=24)
    unixtime = int(time.mktime(last_24.timetuple()))
    return str(unixtime)



