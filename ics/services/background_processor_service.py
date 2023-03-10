import logging
import multiprocessing
import os
import signal
import traceback
from functools import partial
from logging import handlers
from multiprocessing import BoundedSemaphore, Process
from multiprocessing.pool import Pool
from time import sleep

import cherrypy
from sqlalchemy.exc import InvalidRequestError, ResourceClosedError, InterfaceError, InternalError
from sqlalchemy.orm.exc import DetachedInstanceError
from sqlalchemy.pool import NullPool

from ics.db.sqlalchemydb import SQLAlchemyDB, Job

__author__ = 'Andrea Esuli'

LOOP_WAIT = 0.1


def setup_background_processor_log(access_filename, app_filename, environment):
    path = os.path.dirname(access_filename)
    os.makedirs(path, exist_ok=True)
    path = os.path.dirname(app_filename)
    os.makedirs(path, exist_ok=True)

    cherrypy.config.update({
        'environment': environment,
        'log.error_file': '',
        'log.access_file': ''})

    process = multiprocessing.current_process()

    error_handler = handlers.TimedRotatingFileHandler(f'{app_filename}-{process.name.replace(":","_")}.log',
                                                      when='midnight')
    error_handler.setLevel(logging.DEBUG)
    cherrypy.log.error_log.addHandler(error_handler)

    access_handler = handlers.TimedRotatingFileHandler(f'{access_filename}-{process.name.replace(":","_")}.log',
                                                       when='midnight')
    access_handler.setLevel(logging.DEBUG)
    cherrypy.log.access_log.addHandler(access_handler)


class JobError:
    def __init__(self, exception, msg, tb):
        self.e = str(exception)
        self.msg = msg
        self.tb = tb

    def __str__(self):
        return f'{self.__class__.__name__}(\'{self.e}\')\n{self.msg}\n{self.tb}'


process_db = None

MAX_RETRY = 10


def job_launcher(id, f, *args, **kwargs):
    global process_db
    attempt = 0
    while attempt < MAX_RETRY:
        try:
            attempt += 1
            f(*((process_db,) + args), **kwargs)
            break
        except (AttributeError, InternalError, ResourceClosedError, InvalidRequestError, InterfaceError,
                DetachedInstanceError) as e:
            if attempt < MAX_RETRY:
                cherrypy.log(f'FAIL({attempt}) {multiprocessing.current_process().name} {type(e)} {id} {f}',
                             severity=logging.INFO)
            else:
                tb = traceback.format_exc()
                return JobError(e, "FAILED", tb)
        except Exception as e:
            tb = traceback.format_exc()
            return JobError(e, "UNMANAGED", tb)


def bp_pool_initializer(db_connection_string, initializer, *initargs):
    if initializer is not None:
        initializer(*initargs)
    cherrypy.log(f'BackgroundProcessor: adding {multiprocessing.current_process().name} to pool', severity=logging.INFO)
    global process_db
    process_db = SQLAlchemyDB(db_connection_string, NullPool)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class BackgroundProcessor(Process):
    def __init__(self, db_connection_string, pool_size, initializer=None, initargs=None):
        Process.__init__(self)
        self._stop_event = multiprocessing.Event()
        self._pool_size = pool_size
        self._db_connection_string = db_connection_string
        self._initializer = initializer
        self._pool_initializer = partial(bp_pool_initializer, db_connection_string, initializer)
        if initargs is None:
            initargs = []
        self._initargs = initargs
        self._running = False
        self._semaphore = BoundedSemaphore(self._pool_size)

    def run(self):
        self._initializer(*self._initargs)
        with SQLAlchemyDB(self._db_connection_string) as db, \
                Pool(processes=self._pool_size, initializer=self._pool_initializer, initargs=self._initargs) as pool:
            cherrypy.log('BackgroundProcessor: started', severity=logging.INFO)
            while not self._stop_event.is_set():
                try:
                    job = db.get_next_pending_job()
                except InvalidRequestError as ire:
                    cherrypy.log(
                        f'Error fetching next job \nException: {ire}',
                        severity=logging.ERROR)
                    job = None
                if job is None:
                    try:
                        sleep(LOOP_WAIT)
                    finally:
                        continue
                self._semaphore.acquire()
                try:
                    db.set_job_start_time(job.id)
                    db.set_job_status(job.id, Job.status_running)
                    cherrypy.log(f'Starting {job.id}: {job.action["function"]} ({job.description})',
                                 severity=logging.INFO)
                    pool.apply_async(partial(job_launcher, job.id, job.action['function']), job.action['args'],
                                     job.action['kwargs'],
                                     callback=partial(self._release, db, job.id, Job.status_done),
                                     error_callback=partial(self._release, db, job.id, Job.status_error))
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

    def _release(self, db, job_id, status, msg=None):
        try:
            if type(msg) == JobError:
                status = Job.status_error
                cherrypy.log(f'Error {job_id}: {status} Message: {msg}', severity=logging.ERROR)
            else:
                cherrypy.log(f'Completed {job_id} {status} Message: {msg}', severity=logging.INFO)
            db.set_job_completion_time(job_id)
            db.set_job_status(job_id, status)
            if hasattr(msg, 're_raise'):
                msg.re_raise()
        finally:
            self._semaphore.release()

    def version(self):
        import ics
        return ics.__version__
