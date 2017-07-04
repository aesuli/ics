import logging
from functools import partial
from multiprocessing import BoundedSemaphore
from multiprocessing.pool import Pool
from threading import Thread
from time import sleep

import cherrypy

from db.sqlalchemydb import SQLAlchemyDB, Job
from util.util import logged_call

__author__ = 'Andrea Esuli'

LOOP_WAIT = 1


class BackgroundProcessor(Thread):
    def __init__(self, db_connection_string, pool_size=10):
        Thread.__init__(self)
        self._semaphore = BoundedSemaphore(pool_size)
        self._pool = Pool(processes=pool_size)
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
                self._pool.apply_async(job.action['function'], job.action['args'], job.action['kwargs'],
                                       callback=partial(self._release, job.id, Job.status_done),
                                       error_callback=partial(self._release, job.id, Job.status_error))
            except Exception as e:
                self._semaphore.release()
                cherrypy.log('Error on job ' + str(job) + ':' + str(e), severity=logging.ERROR)

    def close(self):
        self.stop()
        self._pool.close()
        self._pool.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @logged_call
    def _release(self, job_id, status, msg):
        try:
            self._db.set_job_completion_time(job_id)
            self._db.set_job_status(job_id, status)
        finally:
            self._semaphore.release()

    def stop(self):
        self._running = False

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def index(self):
        jobslist = list()
        for job in self._db.get_jobs():
            jobinfo = dict()
            jobinfo['id'] = job.id
            jobinfo['description'] = job.description
            jobinfo['creation'] = str(job.creation)
            jobinfo['start'] = str(job.start)
            jobinfo['completion'] = str(job.completion)
            jobinfo['status'] = job.status
            jobslist.append(jobinfo)
        return jobslist

    @cherrypy.expose
    def rerun_job(self, id):
        self._db.set_job_status(id, Job.status_pending)
        return 'Ok'

    @cherrypy.expose
    def delete_job(self, id):
        self._db.delete_job(id)
        return 'Ok'

    @cherrypy.expose
    def delete_all_jobs_done(self):
        to_remove = set()
        for job in self._db.get_jobs():
            if job.status == Job.status_done:
                if len(job.classification_job) == 0:
                    to_remove.add(job.id)
        for id in to_remove:
            self._db.delete_job(id)
        return 'Ok'


if __name__ == "__main__":
    with BackgroundProcessor('sqlite:///%s' % 'test.db') as wcc:
        cherrypy.quickstart(wcc, '/service/bp')
