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
        return "0.2.1 (db: %s)" % self._db.version()


if __name__ == "__main__":
    with JobsService('sqlite:///%s' % 'test.db') as js:
        cherrypy.quickstart(js, '/service/js')
