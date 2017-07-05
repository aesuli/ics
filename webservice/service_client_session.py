import json
from time import sleep

from requests import Session


class ServiceClientSession:
    def __init__(self, protocol, host, port, classifier_path, dataset_path, jobs_path, auth_path):
        self._protocol = protocol
        self._host = host
        self._port = port
        self._classifier_path = classifier_path
        self._dataset_path = dataset_path
        self._jobs_path = jobs_path
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
        url = self._build_url(self._jobs_path)
        r = self._session.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def job_delete(self, id):
        url = self._build_url(self._jobs_path + '/delete_job/')
        r = self._session.post(url, data={'id': id})
        r.raise_for_status()

    def job_rerun(self, id):
        url = self._build_url(self._jobs_path + '/rerun_job/')
        r = self._session.post(url, data={'id': id})
        r.raise_for_status()

    def jobs_delete_all_done(self):
        url = self._build_url(self._jobs_path + '/delete_all_jobs_done/')
        r = self._session.get(url)
        r.raise_for_status()

    def job_completed(self, id):
        url = self._build_url(self._jobs_path + '/job_completed/')
        r = self._session.post(url, data={'id': id})
        r.raise_for_status()
        return json.loads(r.content.decode())

    def wait_for_jobs(self, job_ids, loop_wait=1):
        for job_id in job_ids:
            while not self.job_completed(job_id):
                sleep(loop_wait)

    # classifiers

    def classifiers(self):
        url = self._build_url(self._classifier_path)
        r = self._session.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_create(self, name, labels, overwrite=False):
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
        return json.loads(r.content.decode())

    def classifier_update(self, name, X, y):
        url = self._build_url(self._classifier_path + '/update/')
        r = self._session.post(url, data={'name': name, 'X': X, 'y': y})
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_rename(self, name, new_name):
        url = self._build_url(self._classifier_path + '/rename/')
        r = self._session.post(url, data={'name': name, 'new_name': new_name})
        r.raise_for_status()

    def classifier_labels(self, name):
        url = self._build_url(self._classifier_path + '/labels/' + name)
        r = self._session.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_rename_label(self, classifier_name, label_name, new_name):
        url = self._build_url(self._classifier_path + '/rename_label/')
        r = self._session.post(url, data={'classifier_name': classifier_name, 'label_name': label_name,
                                          'new_name': new_name})
        r.raise_for_status()

    def classifier_download_training_data(self, name, file, chunk_size=2048):
        url = self._build_url(self._classifier_path + '/download_training_data/' + name)
        r = self._session.get(url, stream=True)
        for chunk in r.iter_content(chunk_size=chunk_size, decode_unicode=True):
            if chunk:
                file.write(chunk)

    def classifier_upload_training_data(self, file):
        url = self._build_url(self._classifier_path + '/upload_training_data/')
        files = {'file': file}
        r = self._session.post(url, files=files)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_classify(self, name, X):
        url = self._build_url(self._classifier_path + '/classify/')
        r = self._session.post(url, data={'name': name, 'X': X})
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_score(self, name, X):
        url = self._build_url(self._classifier_path + '/score/')
        r = self._session.post(url, data={'name': name, 'X': X})
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_extract(self, name, labels):
        url = self._build_url(self._classifier_path + '/extract/')
        r = self._session.post(url, data={'name': name, 'labels': labels})
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_combine(self, name, sources):
        url = self._build_url(self._classifier_path + '/combine/')
        r = self._session.post(url, data={'name': name, 'sources': sources})
        r.raise_for_status()
        return json.loads(r.content.decode())

    # datasets

    def datasets(self):
        url = self._build_url(self._dataset_path)
        r = self._session.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_create(self, name):
        url = self._build_url(self._dataset_path + '/create/')
        r = self._session.post(url, data={'name': name})
        r.raise_for_status()

    def dataset_add_document(self, dataset_name, document_name, document_content):
        url = self._build_url(self._dataset_path + '/add_document/')
        r = self._session.post(url, data={'dataset_name': dataset_name, 'document_name': document_name,
                                          'document_content': document_content})
        r.raise_for_status()

    def dataset_delete_document(self, dataset_name, document_name):
        url = self._build_url(self._dataset_path + '/delete_document/')
        r = self._session.post(url, data={'dataset_name': dataset_name, 'document_name': document_name})
        r.raise_for_status()

    def dataset_rename(self, name, newname):
        url = self._build_url(self._dataset_path + '/rename/')
        r = self._session.post(url, data={'name': name, 'newname': newname})
        r.raise_for_status()

    def dataset_delete(self, name):
        url = self._build_url(self._dataset_path + '/delete/')
        r = self._session.post(url, data={'name': name})
        r.raise_for_status()

    def dataset_upload(self, name, file):
        url = self._build_url(self._dataset_path + '/upload/')
        files = {'file': file}
        data = {'name': name}
        r = self._session.post(url, data=data, files=files)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_download(self, name, file, chunk_size=2048):
        url = self._build_url(self._dataset_path + '/download/' + name)
        r = self._session.get(url, stream=True)
        for chunk in r.iter_content(chunk_size=chunk_size, decode_unicode=True):
            if chunk:
                file.write(chunk)

    def dataset_size(self, name):
        url = self._build_url(self._dataset_path + '/size/' + name)
        r = self._session.get(url)
        r.raise_for_status()
        return int(r.content.decode())

    def dataset_classify(self, name, classifiers):
        url = self._build_url(self._dataset_path + '/classify/')
        r = self._session.post(url, data={'name': name, 'classifiers': classifiers})
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_get_classification_jobs(self, name):
        url = self._build_url(self._dataset_path + '/get_classification_jobs/' + name)
        r = self._session.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_download_classification(self, id, file, chunk_size=2048):
        url = self._build_url(self._dataset_path + '/download_classification/' + str(id))
        r = self._session.get(url, stream=True)
        for chunk in r.iter_content(chunk_size=chunk_size, decode_unicode=True):
            if chunk:
                file.write(chunk)

    def dataset_delete_classification(self, id):
        url = self._build_url(self._dataset_path + '/delete_classification/')
        r = self._session.post(url, data={'id': id})
        r.raise_for_status()
