from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from .cron import Processor

def start():
    p = Processor()
    scheduler = BackgroundScheduler()
    scheduler.add_job(p.my_cron_job, 'interval', seconds=10)
    scheduler.start()