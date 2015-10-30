import os
import threading
import time
from requests import Timeout


__author__ = 'Andrea Esuli'


class FileLock(object):
    def __init__(self, filename, timeout=-1):

        self._filename = filename
        self._file = None

        self._timeout = timeout

        self._thread_lock = threading.Lock()

        self._counter = 0


    @property
    def is_locked(self):
        return self._file is not None

    def acquire(self, timeout=None, poll_intervall=1):
        if timeout is None:
            timeout = self._timeout

        with self._thread_lock:
            self._counter += 1

        try:
            start_time = time.time()
            while True:
                with self._thread_lock:
                    if not self.is_locked:
                        self._acquire()

                if self.is_locked:
                    break
                elif 0 <= timeout < time.time() - start_time:
                    raise Timeout(self._filename)
                else:
                    time.sleep(poll_intervall)
        except:
            with self._thread_lock:
                self._counter = max(0, self._counter - 1)
            raise

        class ReturnProxy(object):

            def __init__(self, lock):
                self.lock = lock
                return None

            def __enter__(self):
                return self.lock

            def __exit__(self, exc_type, exc_value, traceback):
                self.lock.release()
                return None

        return ReturnProxy(lock=self)

    def release(self, force=False):
        with self._thread_lock:

            if self.is_locked:
                self._counter -= 1

                if self._counter == 0 or force:
                    self._release()
                    self._counter = 0
        return None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return None

    def __del__(self):
        self.release(force=True)
        return None

    def _acquire(self):
        open_mode = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_TRUNC
        try:
            fd = os.open(self._filename, open_mode)
        except (IOError, OSError):
            pass
        else:
            self._file = fd
        return None

    def _release(self):
        os.close(self._file)
        self._file = None

        try:
            os.remove(self._filename)
        except OSError:
            pass
        return None