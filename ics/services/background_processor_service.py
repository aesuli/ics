import logging
import multiprocessing
import os
import signal
from functools import partial
from logging import handlers
from multiprocessing import BoundedSemaphore, Process
from multiprocessing.pool import Pool
from time import sleep

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


class JobError:
    def __init__(self, exception):
        self.msg = str(exception)

    def __str__(self):
        return f'{self.__class__.__name__}(\'{self.msg}\')'


def ExceptionCatcher(f, *args, **kwargs):
    try:
        f(*args, **kwargs)
    except Exception as e:
        return JobError(e)


def no_sigint(initializer, *initargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if initializer is not None:
        initializer(*initargs)


class BackgroundProcessor(Process):
    def __init__(self, db_connection_string, pool_size, initializer=None, initargs=None):
        Process.__init__(self)
        self._stop_event = multiprocessing.Event()
        self._pool_size = pool_size
        self._initializer = partial(no_sigint, initializer)
        if initargs is None:
            initargs = []
        self._initargs = initargs
        self._db_connection_string = db_connection_string
        self._running = False
        self._semaphore = BoundedSemaphore(self._pool_size)

    def run(self):
        with SQLAlchemyDB(self._db_connection_string) as db, \
                Pool(processes=self._pool_size, initializer=self._initializer, initargs=self._initargs) as pool:
            cherrypy.log('BackgroundProcessor: started', severity=logging.INFO)
            while not self._stop_event.is_set():
                job = db.get_next_pending_job()
                if job is None:
                    try:
                        sleep(LOOP_WAIT)
                    finally:
                        continue
                self._semaphore.acquire()
                try:
                    db.set_job_start_time(job.id)
                    db.set_job_status(job.id, Job.status_running)
                    cherrypy.log('Starting ' + str(job.id) + ': ' + str(job.action['function']), severity=logging.INFO)
                    pool.apply_async(partial(ExceptionCatcher, job.action['function']), job.action['args'],
                                           job.action['kwargs'],
                                           callback=partial(self._release, job.id, Job.status_done),
                                           error_callback=partial(self._release, job.id, Job.status_error))
                except Exception as e:
                    self._semaphore.release()
                    cherrypy.log(
                        'Error on job ' + str(job.id) + ': ' + str(job.action['function']) + '\nException: ' + str(e),
                        severity=logging.ERROR)
            pool.close()
            pool.join()
            cherrypy.log('BackgroundProcessor: stopped', severity=logging.INFO)

    def stop(self):
        self._stop_event.set()
        cherrypy.log('BackgroundProcessor: stopping')
        self.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False

    def _release(self, job_id, status, msg=None):
        try:
            if type(msg) == JobError:
                status = Job.status_error
                cherrypy.log('Error ' + str(job_id) + ' ' + str(status) + ' Message: ' + str(msg),
                             severity=logging.ERROR)
            else:
                cherrypy.log('Completed ' + str(job_id) + ' ' + str(status) + ' Message: ' + str(msg),
                             severity=logging.INFO)
            with SQLAlchemyDB(self._db_connection_string) as db:
                db.set_job_completion_time(job_id)
                db.set_job_status(job_id, status)
            if hasattr(msg, 're_raise'):
                msg.re_raise()
        finally:
            self._semaphore.release()

    def version(self):
        import ics
        return ics.__version__
