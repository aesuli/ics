import logging
import cherrypy
from multiprocessing import Queue, BoundedSemaphore
from multiprocessing.pool import Pool
from threading import Thread
from webservice.util import logged_call


__author__ = 'Andrea Esuli'


class BackgroundProcessor(Thread):
    def __init__(self, pool_size=10):
        Thread.__init__(self)
        self._queue = Queue()
        self._semaphore = BoundedSemaphore(pool_size)
        self._pool = Pool(processes=pool_size)

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
                self._pool.apply_async(obj[1], obj[2], obj[3], callback=self._release,
                                       error_callback=self._error_release)
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
    def _release(self, msg):
        self._semaphore.release()

    @logged_call
    def _error_release(self, msg):
        self._semaphore.release()

    def put(self, function, args=(), kwargs={}):
        self._queue.put(('action', function, args, kwargs))
