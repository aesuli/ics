import getpass
import re
import sys
from argparse import ArgumentParser
from cmd import Cmd
from pprint import pprint

from webservice.service_client_session import ServiceClientSession


def print_exception(fn):
    def wrapped(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            print('Error: ' + str(e))

    return wrapped


class CommandLine(Cmd):
    def __init__(self, protocol, host, port, classifier_path, dataset_path,
                 jobs_path, auth_path):
        self._sc = ServiceClientSession(protocol, host, port, classifier_path, dataset_path, jobs_path, auth_path)
        Cmd.__init__(self)
        self.prompt = '> '

    def do_exit(self, args):
        sys.exit()

    @print_exception
    def do_login(self, args):
        '''
        Login command
        > login username
        Password: <typing of password is hidden>
        Logged in as 'username'
        '''
        username = args
        password = getpass.getpass()
        self._sc.login(username, password)
        print('Logged in as \'%s\'' % username)

    @print_exception
    def do_whoami(self, args):
        '''
        Returns the name of the currently logged in user, or an error if not logged in.
        '''
        print(self._sc.whoami())

    @print_exception
    def do_logout(self, args):
        '''
        Logout command.
        This command does not terminate the command prompt.
        '''
        self._sc.logout()

    @print_exception
    def do_users(self, args):
        '''
        Prints the list of registered usernames
        '''
        users = self._sc.users()
        pprint(users)

    @print_exception
    def do_user_create(self, args):
        '''
        Creates a new user
        > user_create username
        Password: <typing of password is hidden>
        '''
        username = args
        password = getpass.getpass()
        self._sc.user_create(username, password)

    @print_exception
    def do_user_change_password(self, args):
        '''
        Changes password
        > user_change_password username
        Password: <typing of password is hidden>
        '''
        username = args
        password = getpass.getpass()
        self._sc.user_change_password(username, password)

    @print_exception
    def do_user_delete(self, args):
        '''
        Deletes a user
        > user_delete username
        '''
        username = args
        self._sc.user_delete(username)

    # jobs

    @print_exception
    def do_jobs(self, args):
        '''
        Lists jobs
        '''
        jobs = self._sc.jobs()
        pprint(jobs)

    @print_exception
    def do_job_delete(self, args):
        '''
        Deletes a job by its id
        > job_delete 42
        '''
        id = args
        self._sc.job_delete(id)

    @print_exception
    def do_job_rerun(self, args):
        '''
        Reruns a job by its id
        > job_rerun 42
        '''
        id = args
        self._sc.job_rerun(id)

    @print_exception
    def do_jobs_delete_all_done(self, args):
        '''
        Deletes all completed jobs
        '''
        self._sc.jobs_delete_all_done()

    @print_exception
    def do_job_completed(self, args):
        '''
        Checks if a job is completed or not
        > job_completed 42
        True
        '''
        id = args
        self._sc.job_completed(id)

    @print_exception
    def do_wait_for_jobs(self, args):
        '''
        Does not returns until the jobs listed by their id are not completed
        > wait_for_jobs 42 23 56
        '''
        ids = re.split('[,\s]+', args)
        ids = [int(x) for x in ids]
        self._sc.wait_for_jobs(ids)

    # classifiers

    @print_exception
    def do_classifiers(self, args):
        '''
        Lists classifiers
        '''
        classifiers = self._sc.classifiers()
        pprint(classifiers)

    @print_exception
    def do_classifier_create(self, args):
        '''
        Creates a classifier with a set of labels
        > classifier_create name label1 label2 label3
        the -o option at the end overwrites any existing classifier with the same name
        > classifier_create name label1 label2 label3 -o
        '''
        args = re.split('[,\s]+', args)
        name = args[0]
        overwrite = False
        if args[-1] == '-o':
            overwrite = True
            args = args[:-1]
        labels = args
        self._sc.classifier_create(name, labels, overwrite)

    @print_exception
    def do_classifier_delete(self, args):
        '''
        Deletes a classifier
        > classifier_delete username
        '''
        classifier = args
        self._sc.classifier_delete(classifier)

    @print_exception
    def do_classifier_duplicate(self, args):
        '''
        Duplicates a classifier with a new name
        > classifier_create name new_name
        the -o option at the end overwrites any existing classifier with the same new_name
        > classifier_create name new_name -o
        '''
        args = re.split('[,\s]+', args)
        name = args[0]
        new_name = args[1]
        overwrite = False
        if args[-1] == '-o':
            overwrite = True
        self._sc.classifier_duplicate(name, new_name, overwrite)

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--protocol', help='host protocol', type=str, default='http')
    parser.add_argument('--host', help='host server address', type=str, default='127.0.0.1')
    parser.add_argument('--port', help='host server port', type=int, default=8080)
    parser.add_argument('--auth_path', help='server path of the auth web service', type=str, default='/service/auth')
    parser.add_argument('--classifier_path', help='server path of the classifier web service', type=str,
                        default='/service/classifiers')
    parser.add_argument('--dataset_path', help='server path of the dataset web service', type=str,
                        default='/service/datasets')
    parser.add_argument('--jobs_path', help='server path of the jobs web service', type=str,
                        default='/service/jobs')
    args = parser.parse_args(sys.argv[1:])

    command_line = CommandLine(args.protocol, args.host, args.port, args.classifier_path, args.dataset_path,
                               args.jobs_path, args.auth_path)

    command_line.cmdloop('Welcome, type help to have a list of commands')
