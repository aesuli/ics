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

    def user_create(self, username, password):
        url = self._build_url(self._auth_path + '/create_user/')
        r = self._session.post(url, data={'username': username, 'password': password})
        r.raise_for_status()

    def user_change_password(self, username, password):
        url = self._build_url(self._auth_path + '/change_password/')
        r = self._session.post(url, data={'username': username, 'password': password})
        r.raise_for_status()

    def user_delete(self, username):
        url = self._build_url(self._auth_path + '/delete_user/')
        r = self._session.post(url, data={'username': username})
        r.raise_for_status()

    # jobs

    def jobs(self):
        url = self._build_url(self._processor_path)
        r = self._session.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def job_delete(self, id):
        url = self._build_url(self._processor_path + '/delete_job/')
        r = self._session.post(url, data={'id': id})
        r.raise_for_status()

    def jobs_delete_all_done(self):
        url = self._build_url(self._processor_path + '/delete_all_jobs_done/')
        r = self._session.get(url)
        r.raise_for_status()

    # classifiers

    def classifiers(self):
        url = self._build_url(self._classifier_path)
        r = self._session.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_create(self, name, labels, overwrite = False):
        url = self._build_url(self._classifier_path + '/create/')
        r = self._session.post(url, data={'name': name, 'labels': labels, 'overwrite': overwrite})
        r.raise_for_status()

    def classifier_delete(self, name):
        url = self._build_url(self._classifier_path + '/delete/')
        r = self._session.post(url, data={'name': name})
        r.raise_for_status()

    def classifier_duplicate(self, name, new_name, overwrite=False):
        url = self._build_url(self._classifier_path + '/duplicate/')
        r = self._session.post(url, data={'name': name, 'new_name': new_name, 'overwrite': overwrite})
        r.raise_for_status()

    def classifier_update(self,name, X, y):
        url = self._build_url(self._classifier_path + '/update/')
        r = self._session.post(url, data={'name': name, 'X': X, 'y': y})
        r.raise_for_status()

    def classifier_rename(self, name, new_name):
        url = self._build_url(self._classifier_path + '/rename/')
        r = self._session.post(url, data={'name': name, 'new_name': new_name})
        r.raise_for_status()

    def classifier_labels(self, name):
        url = self._build_url(self._classifier_path + '/labels/'+name)
        r = self._session.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_rename_label(self, classifier_name, label_name, new_name):
        url = self._build_url(self._classifier_path + '/rename_label/')
        r = self._session.post(url, data={'classifier_name': classifier_name, 'label_name': label_name, 'new_name': new_name})
        r.raise_for_status()

    def classifier_download_training_data(self, name, file, chunk_size = 2048):
        url = self._build_url(self._classifier_path + '/download_training_data/'+name)
        r = self._session.get(url, stream=True)
        for chunk in r.iter_content(chunk_size=chunk_size, decode_unicode=True):
            if chunk:
                file.write(chunk)

    def classifier_upload_training_data(self, file):
        url = self._build_url(self._classifier_path + '/upload_training_data/')
        files = {'file': file}
        r = self._session.post(url,files=files)
        r.raise_for_status()

    def classifier_classify(self, **data):
        # TODO
        pass

    def classifier_score(self, **data):
        # TODO
        pass

    def classifier_extract(self, **data):
        # TODO
        pass

    def classifier_combine(self, **data):
        # TODO
        pass

    # datasets

    def datasets(self):
        url = self._build_url(self._dataset_path)
        r = self._session.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode())
