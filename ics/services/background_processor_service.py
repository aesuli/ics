import logging
import multiprocessing
import os
import sys
from functools import partial
from logging import handlers
from multiprocessing import BoundedSemaphore
from multiprocessing.pool import Pool
from threading import Thread
from time import sleep
import tblib.pickling_support
tblib.pickling_support.install()

import cherrypy

from ics.db.sqlalchemydb import SQLAlchemyDB, Job

__author__ = 'Andrea Esuli'

LOOP_WAIT = 0.1

def setup_background_processor_log(access_filename, app_filename):
    path = os.path.dirname(access_filename)
    os.makedirs(path, exist_ok=True)
    path = os.path.dirname(app_filename)
    os.makedirs(path, exist_ok=True)

    cherrypy.config.update({  # 'environment': 'production',
        'log.error_file': '',
        'log.access_file': ''})

    process = multiprocessing.current_process()

    error_handler = handlers.TimedRotatingFileHandler(app_filename + '-' + str(process.name) + '.log',
                                                      when='midnight')
    error_handler.setLevel(logging.DEBUG)
    cherrypy.log.error_log.addHandler(error_handler)

    access_handler = handlers.TimedRotatingFileHandler(access_filename + '-' + str(process.name) + '.log',
                                                       when='midnight')
    access_handler.setLevel(logging.DEBUG)
    cherrypy.log.access_log.addHandler(access_handler)

class ExceptionWrapper(object):

    def __init__(self, ee):
        self.ee = ee
        __, __, self.tb = sys.exc_info()

    def re_raise(self):
        raise self.ee.with_traceback(self.tb)

def ExceptionCatcher(f, *args, **kwargs):
    try:
        f(*args,**kwargs)
    except Exception as e:
        return ExceptionWrapper(e)

class BackgroundProcessor(Thread):
    def __init__(self, db_connection_string, pool_size, initializer=None, initargs=None):
        Thread.__init__(self)
        self._semaphore = BoundedSemaphore(pool_size)
        self._pool = Pool(processes=pool_size, initializer=initializer, initargs=initargs)
        self._db = SQLAlchemyDB(db_connection_string)
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            job = self._db.get_next_pending_job()
            if job is None:
                sleep(LOOP_WAIT)
                continue
            self._semaphore.acquire()
            try:
                self._db.set_job_start_time(job.id)
                self._db.set_job_status(job.id, Job.status_running)
                cherrypy.log('Starting ' + str(job.id) + ': ' + str(job.action['function']), severity=logging.INFO)
                self._pool.apply_async(partial(ExceptionCatcher,job.action['function']), job.action['args'], job.action['kwargs'],
                                       callback=partial(self._release, job.id, Job.status_done),
                                       error_callback=partial(self._release, job.id, Job.status_error))
            except Exception as e:
                self._semaphore.release()
                cherrypy.log(
                    'Error on job ' + str(job.id) + ': ' + str(job.action['function']) + '\nException: ' + str(e),
                    severity=logging.ERROR)

    def close(self):
        self.stop()
        self._db.close()
        self._pool.close()
        self._pool.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _release(self, job_id, status, msg=None):
        try:
            if hasattr(msg,'re_raise'):
                status = Job.status_error
            cherrypy.log('Completed ' + str(job_id) + ' ' + str(status) + ' Message: ' + str(msg),
                         severity=logging.INFO)
            self._db.set_job_completion_time(job_id)
            self._db.set_job_status(job_id, status)
            if hasattr(msg,'re_raise'):
                msg.re_raise()
        finally:
            self._semaphore.release()

    def stop(self):
        self._running = False

    def version(self):
        import ics
        return ics.__version__
