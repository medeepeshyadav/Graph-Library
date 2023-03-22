from django.apps import AppConfig
import schedule
import time
# from .cron import Processor

class GraphAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'graph_app'

    def ready(self):
        from jobScheduler import job_scheduler
        job_scheduler.start()

