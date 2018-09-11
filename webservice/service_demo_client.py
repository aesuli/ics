import json

import requests
from requests import HTTPError

__author__ = 'Andrea Esuli'


class ServiceDemoClient:
    def __init__(self, protocol, host, port, classifier_path, ip_auth_path, key_auth_path):
        self._protocol = protocol
        self._host = host
        if type(port) == int:
            port = str(port)
        self._port = port
        self._classifier_path = classifier_path
        self._ip_auth_path = ip_auth_path
        self._key_auth_path = key_auth_path

    def _build_url(self, path):
        return self._protocol + '://' + self._host + ':' + self._port + '/' + path

    def classifiers(self):
        url = self._build_url(self._classifier_path + '/info/')
        r = requests.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def classifier_classify(self, name, X, authkey=None):
        url = self._build_url(self._classifier_path + '/classify/')
        if authkey:
            data = {'name': name, 'X': X, 'authkey': authkey}
        else:
            data = {'name': name, 'X': X}
        r = requests.post(url, data=data)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def ip_info(self):
        url = self._build_url(self._ip_auth_path + '/info/')
        r = requests.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode())

    def key_info(self, authkey):
        url = self._build_url(self._key_auth_path + '/info/')
        data = {'authkey': authkey}
        r = requests.post(url, data)
        r.raise_for_status()
        return json.loads(r.content.decode())


if __name__ == '__main__':
    protocol = 'http'
    host = 'label.esuli.it'
    port = 80
    classifier_path = 'service/classifiers'
    ip_auth_path = 'service/ipauth'
    key_auth_path = 'service/keyauth'

    key = 'you_key_here'

    service_client = ServiceDemoClient(protocol, host, port, classifier_path, ip_auth_path, key_auth_path)

    classifiers = service_client.classifiers()

    test_texts = ['this is a text', 'this is another one']

    print()
    print('Documents to be labeled:', test_texts)
    for classifier in classifiers:
        print()
        print('Using classifier:', classifier)
        print()
        print('Assigned labels:', service_client.classifier_classify(classifier['name'], test_texts))

    print()
    print('IP requests stats:', service_client.ip_info())

    try:
        print()
        print('Key requests stats:', service_client.key_info(key))
    except HTTPError:
        print('Error: the key stats method works only with valid keys')
