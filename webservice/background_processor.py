from functools import partial
import logging
from multiprocessing import Queue, BoundedSemaphore
from multiprocessing.pool import Pool
from threading import Thread

import cherrypy
from db.sqlalchemydb import SQLAlchemyDB

from util.util import logged_call


__author__ = 'Andrea Esuli'


class BackgroundProcessor(Thread):
    def __init__(self, db_connection_string, pool_size=10):
        Thread.__init__(self)
        self._queue = Queue()
        self._semaphore = BoundedSemaphore(pool_size)
        self._pool = Pool(processes=pool_size)
        self._db = SQLAlchemyDB(db_connection_string)

    def run(self):
        while True:
            obj = self._queue.get()
            self._semaphore.acquire()
            if obj[0] == 'stop':
                self._pool.close()
                self._pool.join()
                self._semaphore.release()
                break
            elif obj[0] == 'action':
                try:
                    job_id = obj[1]
                    self._db.set_job_start_time(job_id)
                    self._db.set_job_status(job_id, 'running')
                    self._pool.apply_async(obj[2], obj[3], obj[4], callback=partial(self._release,job_id),
                                           error_callback=partial(self._error_release,job_id))
                except:
                    self._semaphore.release()
            else:
                cherrypy.log('Unknown request ' + str(obj), severity=logging.ERROR)
                self._semaphore.release()

    def close(self):
        self._queue.empty()
        self._queue.put(('stop',))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @logged_call
    def _release(self, job_id, msg):
        self._db.set_job_completion_time(job_id)
        self._db.set_job_status(job_id, 'done')
        self._semaphore.release()

    @logged_call
    def _error_release(self, job_id, msg):
        self._db.set_job_completion_time(job_id)
        self._db.set_job_status(job_id, 'error')
        self._semaphore.release()

    def put(self, function, args=(), kwargs={}, description=None):
        if description is None:
            description = function.__name__
        job_id = self._db.create_job(description)
        self._queue.put(('action', job_id, function, args, kwargs))
