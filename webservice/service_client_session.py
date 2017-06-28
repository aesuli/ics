import json

from requests import Session


class ServiceClientSession:
    def __init__(self, protocol, host, port, classifier_path, dataset_path, processor_path, auth_path):
        self._protocol = protocol
        self._host = host
        self._port = port
        self._classifier_path = classifier_path
        self._dataset_path = dataset_path
        self._processor_path = processor_path
        self._auth_path = auth_path
        self._session = Session()

    def _build_url(self, path):
        return self._protocol + '://' + self._host + ':' + self._port + '/' + path

    # auth

    def login(self, username, password):
        url = self._build_url(self._auth_path + '/login/')
        r = self._session.post(url, data={'username': username, 'password': password})
        r.raise_for_status()

    def logout(self):
        url = self._build_url(self._auth_path + '/logout/')
        r = self._session.get(url)
        r.raise_for_status()

    def users(self):
        url = self._build_url(self._auth_path)
        r = self._session.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def create_user(self, username, password):
        url = self._build_url(self._auth_path + '/create_user/')
        r = self._session.post(url, data={'username': username, 'password': password})
        r.raise_for_status()

    def change_password(self, username, password):
        url = self._build_url(self._auth_path + '/change_password/')
        r = self._session.post(url, data={'username': username, 'password': password})
        r.raise_for_status()

    def delete_user(self, username):
        url = self._build_url(self._auth_path + '/delete_user/')
        r = self._session.post(url, data={'username': username})
        r.raise_for_status()

    #jobs

    def jobs(self):
        url = self._build_url(self._processor_path)
        r = self._session.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def delete_job(self, id):
        url = self._build_url(self._processor_path + '/delete_job/')
        r = self._session.post(url, data={'id': id})
        r.raise_for_status()

    def delete_all_jobs_done(self):
        url = self._build_url(self._processor_path + '/delete_all_jobs_done/')
        r = self._session.get(url)
        r.raise_for_status()

    #classifiers

    def classifiers(self):
        url = self._build_url(self._classifier_path)
        r = self._session.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode())

    #datasets
