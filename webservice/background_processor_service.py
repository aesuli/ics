import logging
from functools import partial
from multiprocessing import BoundedSemaphore
from multiprocessing.pool import Pool
from threading import Thread
from time import sleep

import cherrypy

from db.sqlalchemydb import SQLAlchemyDB, Job

__author__ = 'Andrea Esuli'

LOOP_WAIT = 0.1


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
                self._pool.apply_async(job.action['function'], job.action['args'], job.action['kwargs'],
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

    def _release(self, job_id, status, msg):
        try:
            cherrypy.log('Completed ' + str(job_id) + ' ' + str(status))
            self._db.set_job_completion_time(job_id)
            self._db.set_job_status(job_id, status)
        finally:
            self._semaphore.release()

    def stop(self):
        self._running = False

    def version(self):
        return "0.3.1 (db: %s)" % self._db.version()
