import json
from time import sleep

from requests import Session, HTTPError

__author__ = 'Andrea Esuli'


class ClientSession:
    def __init__(self, protocol, host, port, classifier_path=None, dataset_path=None, jobs_path=None,
                 user_auth_path=None, key_auth_path=None, ip_auth_path=None):
        self._protocol = protocol
        self._host = host
        if type(port) == int:
            port = str(port)
        self._port = port
        self._classifier_path = classifier_path
        self._dataset_path = dataset_path
        self._jobs_path = jobs_path
        self._user_auth_path = user_auth_path
        self._key_auth_path = key_auth_path
        self._ip_auth_path = ip_auth_path
        self._session = Session()
        self._default_data = dict()

    def _build_url(self, path):
        return self._protocol + '://' + self._host + ':' + self._port + '' + path

    def _get_default_post_data(self):
        return dict(self._default_data)

    # user auth

    def login(self, username, password):
        url = self._build_url(self._user_auth_path + '/login/')
        data = self._get_default_post_data()
        data['name'] = username
        data['password'] = password
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def whoami(self):
        url = self._build_url(self._user_auth_path + '/whoami/')
        data = self._get_default_post_data()
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return r.content.decode()

    def logout(self):
        url = self._build_url(self._user_auth_path + '/logout/')
        data = self._get_default_post_data()
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def user_info(self, page=None, page_size=50):
        url = self._build_url(self._user_auth_path + '/info/')
        data = self._get_default_post_data()
        data['page'] = page
        data['page_size'] = page_size
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def user_count(self):
        url = self._build_url(self._user_auth_path + '/count/')
        data = self._get_default_post_data()
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def user_create(self, username, password, hourly_limit, request_limit):
        url = self._build_url(self._user_auth_path + '/create/')
        data = self._get_default_post_data()
        data['name'] = username
        data['password'] = password
        data['hourly_limit'] = hourly_limit
        data['request_limit'] = request_limit
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def user_change_password(self, username, password):
        url = self._build_url(self._user_auth_path + '/change_password/')
        data = self._get_default_post_data()
        data['name'] = username
        data['password'] = password
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def user_delete(self, username):
        url = self._build_url(self._user_auth_path + '/delete/')
        data = self._get_default_post_data()
        data['name'] = username
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def user_set_hourly_limit(self, username, hourly_limit):
        url = self._build_url(self._user_auth_path + '/set_hourly_limit/')
        data = self._get_default_post_data()
        data['name'] = username
        data['hourly_limit'] = hourly_limit
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def user_set_request_limit(self, username, request_limit):
        url = self._build_url(self._user_auth_path + '/set_request_limit/')
        data = self._get_default_post_data()
        data['name'] = username
        data['request_limit'] = request_limit
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def user_set_current_request_counter(self, username, count=0):
        url = self._build_url(self._user_auth_path + '/set_current_request_counter/')
        data = self._get_default_post_data()
        data['name'] = username
        data['count'] = count
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def user_version(self):
        url = self._build_url(self._user_auth_path + '/version/')
        data = self._get_default_post_data()
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    # key auth

    def key_info(self, page=None, page_size=50):
        url = self._build_url(self._key_auth_path + '/info/')
        data = self._get_default_post_data()
        data['page'] = page
        data['page_size'] = page_size
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def key_count(self):
        url = self._build_url(self._key_auth_path + '/count/')
        data = self._get_default_post_data()
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def key_use(self, key):
        self._default_data['authkey'] = key
        return 'Ok'

    def key_disable_use(self):
        try:
            del self._default_data['authkey']
        except KeyError:
            pass
        return 'Ok'

    def key_create(self, name, hourly_limit, request_limit):
        url = self._build_url(self._key_auth_path + '/create/')
        data = self._get_default_post_data()
        data['name'] = name
        data['hourly_limit'] = hourly_limit
        data['request_limit'] = request_limit
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def key_delete(self, key):
        url = self._build_url(self._key_auth_path + '/delete/')
        data = self._get_default_post_data()
        data['key'] = key
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def key_set_hourly_limit(self, key, hourly_limit):
        url = self._build_url(self._key_auth_path + '/set_hourly_limit/')
        data = self._get_default_post_data()
        data['key'] = key
        data['hourly_limit'] = hourly_limit
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def key_set_request_limit(self, key, request_limit):
        url = self._build_url(self._key_auth_path + '/set_request_limit/')
        data = self._get_default_post_data()
        data['key'] = key
        data['request_limit'] = request_limit
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def key_set_current_request_counter(self, key, count=0):
        url = self._build_url(self._key_auth_path + '/set_current_request_counter/')
        data = self._get_default_post_data()
        data['key'] = key
        data['count'] = count
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def key_version(self):
        url = self._build_url(self._key_auth_path + '/version/')
        data = self._get_default_post_data()
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    # ip auth

    def ip_info(self, page=None, page_size=50):
        url = self._build_url(self._ip_auth_path + '/info/')
        data = self._get_default_post_data()
        data['page'] = page
        data['page_size'] = page_size
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def ip_count(self):
        url = self._build_url(self._ip_auth_path + '/count/')
        data = self._get_default_post_data()
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def ip_create(self, ip, hourly_limit, request_limit):
        url = self._build_url(self._ip_auth_path + '/create/')
        data = self._get_default_post_data()
        data['ip'] = ip
        data['hourly_limit'] = hourly_limit
        data['request_limit'] = request_limit
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def ip_delete(self, ip):
        url = self._build_url(self._ip_auth_path + '/delete/')
        data = self._get_default_post_data()
        data['ip'] = ip
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def ip_set_hourly_limit(self, ip, hourly_limit):
        url = self._build_url(self._ip_auth_path + '/set_hourly_limit/')
        data = self._get_default_post_data()
        data['ip'] = ip
        data['hourly_limit'] = hourly_limit
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def ip_set_request_limit(self, ip, request_limit):
        url = self._build_url(self._ip_auth_path + '/set_request_limit/')
        data = self._get_default_post_data()
        data['ip'] = ip
        data['request_limit'] = request_limit
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def ip_set_current_request_counter(self, ip, count=0):
        url = self._build_url(self._ip_auth_path + '/set_current_request_counter/')
        data = self._get_default_post_data()
        data['ip'] = ip
        data['count'] = count
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def ip_version(self):
        url = self._build_url(self._ip_auth_path + '/version/')
        data = self._get_default_post_data()
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    # jobs

    def job_info(self, page=None, page_size=50):
        url = self._build_url(self._jobs_path + '/info/')
        data = self._get_default_post_data()
        data['page'] = page
        data['page_size'] = page_size
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def job_count(self):
        url = self._build_url(self._jobs_path + '/count/')
        data = self._get_default_post_data()
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def job_delete(self, id):
        url = self._build_url(self._jobs_path + '/delete/')
        data = self._get_default_post_data()
        data['id'] = str(id)
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def job_rerun(self, id):
        url = self._build_url(self._jobs_path + '/rerun/')
        data = self._get_default_post_data()
        data['id'] = str(id)
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def job_delete_all_done(self):
        url = self._build_url(self._jobs_path + '/delete_all_done/')
        data = self._get_default_post_data()
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def job_completed(self, id):
        url = self._build_url(self._jobs_path + '/completed/')
        data = self._get_default_post_data()
        data['id'] = str(id)
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def job_wait(self, job_ids, loop_wait=1):
        for job_id in job_ids:
            while not self.job_completed(job_id):
                sleep(loop_wait)
        return 'Ok'

    def job_lock_info(self, page=None, page_size=50):
        url = self._build_url(self._jobs_path + '/lock_info/')
        data = self._get_default_post_data()
        data['page'] = page
        data['page_size'] = page_size
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def job_lock_count(self):
        url = self._build_url(self._jobs_path + '/lock_count/')
        data = self._get_default_post_data()
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def job_lock_delete(self, name):
        url = self._build_url(self._jobs_path + '/lock_delete/')
        data = self._get_default_post_data()
        data['name'] = name
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def job_version(self):
        url = self._build_url(self._jobs_path + '/version/')
        data = self._get_default_post_data()
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    # classifiers

    def classifier_info(self, page=None, page_size=50):
        url = self._build_url(self._classifier_path + '/info/')
        data = self._get_default_post_data()
        data['page'] = page
        data['page_size'] = page_size
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_count(self):
        url = self._build_url(self._classifier_path + '/count/')
        data = self._get_default_post_data()
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_types(self):
        url = self._build_url(self._classifier_path + '/classifier_types/')
        data = self._get_default_post_data()
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_create(self, name, labels, type, overwrite=False):
        url = self._build_url(self._classifier_path + '/create/')
        data = self._get_default_post_data()
        data['name'] = name
        data['labels'] = labels
        data['type'] = type
        data['overwrite'] = overwrite
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_delete(self, name):
        url = self._build_url(self._classifier_path + '/delete/')
        data = self._get_default_post_data()
        data['name'] = name
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_update(self, name, X, y):
        url = self._build_url(self._classifier_path + '/update/')
        data = self._get_default_post_data()
        data['name'] = name
        data['X'] = X
        data['y'] = y
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_set_description(self, name, description):
        url = self._build_url(self._classifier_path + '/set_description/')
        data = self._get_default_post_data()
        data['name'] = name
        data['description'] = description
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_rename(self, name, new_name, overwrite=False):
        url = self._build_url(self._classifier_path + '/rename/')
        data = self._get_default_post_data()
        data['name'] = name
        data['new_name'] = new_name
        data['overwrite'] = overwrite
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_label_info(self, name):
        url = self._build_url(self._classifier_path + '/label_info/')
        data = self._get_default_post_data()
        data['name'] = name
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_label_rename(self, name, label_name, new_label_name):
        url = self._build_url(self._classifier_path + '/label_rename/')
        data = self._get_default_post_data()
        data['name'] = name
        data['label_name'] = label_name
        data['new_label_name'] = new_label_name
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_download_training_data(self, name, file, chunk_size=2048):
        url = self._build_url(self._classifier_path + '/download_training_data/')
        data = self._get_default_post_data()
        data['name'] = name
        r = self._session.post(url, data, stream=True)
        for chunk in r.iter_content(chunk_size=chunk_size, decode_unicode=True):
            if chunk:
                file.write(chunk)
        return 'Ok'

    def classifier_download_model(self, name, file, chunk_size=2048):
        url = self._build_url(self._classifier_path + '/download_model/')
        data = self._get_default_post_data()
        data['name'] = name
        r = self._session.post(url, data, stream=True)
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
        return 'Ok'

    def classifier_upload_training_data(self, file, type):
        url = self._build_url(self._classifier_path + '/upload_training_data/')
        data = self._get_default_post_data()
        data['type'] = type
        files = {'file': file}
        r = self._session.post(url, data=data, files=files)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_upload_model(self, name, file, overwrite=False):
        url = self._build_url(self._classifier_path + '/upload_model/')
        data = self._get_default_post_data()
        data['name'] = name
        data['overwrite'] = overwrite
        files = {'file': file}
        r = self._session.post(url, data=data, files=files)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_classify(self, name, X):
        url = self._build_url(self._classifier_path + '/classify/')
        data = self._get_default_post_data()
        data['name'] = name
        data['X'] = X
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_score(self, name, X):
        url = self._build_url(self._classifier_path + '/score/')
        data = self._get_default_post_data()
        data['name'] = name
        data['X'] = X
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_extract(self, name, labels, type):
        url = self._build_url(self._classifier_path + '/extract/')
        data = self._get_default_post_data()
        data['name'] = name
        data['labels'] = labels
        data['type'] = type
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_merge(self, name, sources, type, binary_by_name, overwrite=False):
        url = self._build_url(self._classifier_path + '/merge/')
        data = self._get_default_post_data()
        data['name'] = name
        data['sources'] = sources
        data['type'] = type
        data['binary_by_name'] = binary_by_name
        data['overwrite'] = overwrite
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_version(self):
        url = self._build_url(self._classifier_path + '/version/')
        data = self._get_default_post_data()
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    # datasets

    def dataset_info(self, page=None, page_size=50):
        url = self._build_url(self._dataset_path + '/info/')
        data = self._get_default_post_data()
        data['page'] = page
        data['page_size'] = page_size
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_count(self):
        url = self._build_url(self._dataset_path + '/count/')
        data = self._get_default_post_data()
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_create(self, name):
        url = self._build_url(self._dataset_path + '/create/')
        data = self._get_default_post_data()
        data['name'] = name
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_add_document(self, dataset_name, document_name, document_content):
        url = self._build_url(self._dataset_path + '/add_document/')
        data = self._get_default_post_data()
        data['name'] = dataset_name
        data['document_name'] = document_name
        data['document_content'] = document_content
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_delete_document(self, dataset_name, document_name):
        url = self._build_url(self._dataset_path + '/delete_document/')
        data = self._get_default_post_data()
        data['name'] = dataset_name
        data['document_name'] = document_name
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_rename(self, name, newname):
        url = self._build_url(self._dataset_path + '/rename/')
        data = self._get_default_post_data()
        data['name'] = name
        data['newname'] = newname
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_delete(self, name):
        url = self._build_url(self._dataset_path + '/delete/')
        data = self._get_default_post_data()
        data['name'] = name
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_upload(self, name, file):
        url = self._build_url(self._dataset_path + '/upload/')
        files = {'file': file}
        data = self._get_default_post_data()
        data['name'] = name
        r = self._session.post(url, data=data, files=files)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_download(self, name, file, chunk_size=2048):
        url = self._build_url(self._dataset_path + '/download/')
        data = self._get_default_post_data()
        data['name'] = name
        r = self._session.post(url, data=data, stream=True)
        for chunk in r.iter_content(chunk_size=chunk_size, decode_unicode=True):
            if chunk:
                file.write(chunk)
        return 'Ok'

    def dataset_size(self, name):
        url = self._build_url(self._dataset_path + '/size/')
        data = self._get_default_post_data()
        data['name'] = name
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_document_by_name(self, dataset_name, document_name):
        url = self._build_url(self._dataset_path + '/document_by_name/')
        data = self._get_default_post_data()
        data['name'] = dataset_name
        data['document_name'] = document_name
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_document_by_position(self, name, position):
        url = self._build_url(self._dataset_path + '/document_by_position/')
        data = self._get_default_post_data()
        data['name'] = name
        data['position'] = position
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_most_uncertain_document_id(self, dataset_name, classifier_name):
        url = self._build_url(self._dataset_path + '/most_uncertain_document_id/')
        data = self._get_default_post_data()
        data['name'] = dataset_name
        data['classifier_name'] = classifier_name
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_most_certain_document_id(self, dataset_name, classifier_name):
        url = self._build_url(self._dataset_path + '/most_certain_document_id/')
        data = self._get_default_post_data()
        data['name'] = dataset_name
        data['classifier_name'] = classifier_name
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_classify(self, name, classifiers):
        url = self._build_url(self._dataset_path + '/classify/')
        data = self._get_default_post_data()
        data['name'] = name
        data['classifiers'] = classifiers
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_classification_info(self, name, page=None, page_size=50):
        url = self._build_url(self._dataset_path + '/classification_info/')
        data = self._get_default_post_data()
        data['name'] = name
        data['page'] = page
        data['page_size'] = page_size
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def datatset_classification_count(self, name):
        url = self._build_url(self._dataset_path + '/classification_count/')
        data = self._get_default_post_data()
        data['name'] = name
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_classification_download(self, id, file, chunk_size=2048):
        url = self._build_url(self._dataset_path + '/classification_download/')
        data = self._get_default_post_data()
        data['id'] = str(id)
        r = self._session.post(url, data=data, stream=True)
        for chunk in r.iter_content(chunk_size=chunk_size, decode_unicode=True):
            if chunk:
                file.write(chunk)
        return 'Ok'

    def dataset_classification_delete(self, id):
        url = self._build_url(self._dataset_path + '/classification_delete/')
        data = self._get_default_post_data()
        data['id'] = str(id)
        r = self._session.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def dataset_version(self):
        url = self._build_url(self._dataset_path + '/version/')
        data = self._get_default_post_data()
        r = self._session.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def version(self):
        return "0.3.2"


if __name__ == '__main__':
    protocol = 'http'
    host = 'label.esuli.it'
    port = 80
    classifier_path = 'service/classifiers'
    dataset_path = 'service/datasets'
    jobs_path = 'service/jobs'
    user_auth_path = 'service/userauth'
    ip_auth_path = 'service/ipauth'
    key_auth_path = 'service/keyauth'

    key = 'you_key_here'

    client = ClientSession(protocol, host, port, classifier_path, dataset_path, jobs_path, user_auth_path,
                           key_auth_path, ip_auth_path)

    classifiers = client.classifier_info()

    test_texts = ['this is a text', 'this is another one']

    print()
    print('Documents to be labeled:', test_texts)
    for classifier in classifiers:
        print()
        print('Using classifier:', classifier)
        print()
        print('Assigned labels:', client.classifier_classify(classifier['name'], test_texts))

    print()
    print('IP requests stats:', client.ip_info())

    try:
        print()
        print('Key requests stats:', client.key_info(key))
    except HTTPError:
        print('Error: the key stats method works only with valid keys')
