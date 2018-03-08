import cherrypy

from db.sqlalchemydb import SQLAlchemyDB, Job

__author__ = 'Andrea Esuli'


class JobsService(object):
    def __init__(self, db_connection_string, pool_size=10):
        self._db = SQLAlchemyDB(db_connection_string)

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def info(self, page=0, page_size=50):
        page = max(0,int(page))
        page_size = max(1,int(page_size))
        jobslist = list()
        jobs = self._db.get_jobs()[page * page_size:(page + 1) * page_size]
        for job in jobs:
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
    def count(self):
        return str(len(list(self._db.get_jobs())))

    @cherrypy.expose
    def rerun_job(self, id):
        self._db.set_job_start_time(id, None)
        self._db.set_job_completion_time(id, None)
        self._db.set_job_status(id, Job.status_pending)
        return 'Ok'

    @cherrypy.expose
    def delete_job(self, id):
        self._db.delete_job(id)
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def locks(self, page=0, page_size=50):
        page = max(0,int(page))
        page_size = max(1,int(page_size))
        lockslist = list()
        locks = self._db.get_locks()[page * page_size:(page + 1) * page_size]
        for lock in locks:
            lockinfo = dict()
            lockinfo['name'] = lock.name
            lockinfo['locker'] = lock.locker
            lockinfo['creation'] = str(lock.creation)
            lockslist.append(lockinfo)
        return lockslist

    @cherrypy.expose
    def locks_count(self):
        return str(len(list(self._db.get_locks())))

    @cherrypy.expose
    def delete_lock(self, name):
        self._db.delete_lock(name)
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def job_completed(self, id):
        job_status = self._db.get_job_status(id)
        return job_status == Job.status_done or job_status == Job.status_error or job_status == Job.status_missing

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

    @cherrypy.expose
    def version(self):
        return "0.4.1 (db: %s)" % self._db.version()
