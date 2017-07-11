import getpass
import re
import sys
from argparse import ArgumentParser
from cmd import Cmd
from os import remove
from os.path import exists
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

    def emptyline(self):
        pass

    def help_exit(self):
        print('Exit')

    def do_exit(self, args):
        sys.exit()

    def help_login(self):
        print('''
        Login command
        > login username
        Password: <typing of password is hidden>
        Logged in as \'username\'
        ''')

    @print_exception
    def do_login(self, args):
        username = args.strip()
        if len(username) == 0:
            username = input('Username: ').strip()
        password = getpass.getpass('Password: ')
        self._sc.login(username, password)
        print('Logged in as \'%s\'' % username)

    def help_whoami(self):
        print('''
        Returns the name of the currently logged in user, or an error if not logged in.
        ''')

    @print_exception
    def do_whoami(self, args):
        print(self._sc.whoami())

    def help_logout(self):
        print('''
        Logout command.
        This command does not terminate the command prompt.
        ''')

    @print_exception
    def do_logout(self, args):
        self._sc.logout()

    def help_users(self):
        print('''
        Prints the list of registered usernames
        ''')

    @print_exception
    def do_users(self, args):
        users = self._sc.users()
        pprint(users)

    def help_user_create(self):
        print('''
        Creates a new user
        > user_create username
        Password: <typing of password is hidden>
        ''')

    @print_exception
    def do_user_create(self, args):
        username = args.strip()
        if len(username) == 0:
            username = input('Username: ').strip()
        password = getpass.getpass('Password:')
        confirmpassword = getpass.getpass('Confirm assword:')
        if password == confirmpassword:
            self._sc.user_create(username, password)
        else:
            print('Passwords do not match')

    def help_user_change_password(self):
        print('''
        Changes password
        > user_change_password username
        Password: <typing of password is hidden>
        ''')

    @print_exception
    def do_user_change_password(self, args):
        username = args.strip()
        if len(username) == 0:
            username = input('Username: ').strip()
        password = getpass.getpass('Password:')
        confirmpassword = getpass.getpass('Confirm assword:')
        if password == confirmpassword:
            self._sc.user_change_password(username, password)
        else:
            print('Passwords do not match')

    def help_user_delete(self):
        print('''
        Deletes a user
        > user_delete username
        ''')

    @print_exception
    def do_user_delete(self, args):
        username = args.strip()
        self._sc.user_delete(username)

    # jobs

    def help_jobs(self):
        print('''
        Lists jobs
        ''')

    @print_exception
    def do_jobs(self, args):
        jobs = self._sc.jobs()
        pprint(jobs)

    def help_job_delete(self):
        print('''
        Deletes a job by its id
        > job_delete 42
        ''')

    @print_exception
    def do_job_delete(self, args):
        id = args.strip()
        self._sc.job_delete(id)

    def help_job_rerun(self):
        print('''
        Reruns a job by its id
        > job_rerun 42
        ''')

    @print_exception
    def do_job_rerun(self, args):
        id = args.strip()
        self._sc.job_rerun(id)

    def help_jobs_delete_all_done(self):
        print('''
        Deletes all completed jobs
        ''')

    @print_exception
    def do_jobs_delete_all_done(self, args):
        self._sc.jobs_delete_all_done()

    def help_job_completed(self):
        print('''
        Checks if a job is completed or not
        > job_completed 42
        True
        ''')

    @print_exception
    def do_job_completed(self, args):
        id = args.strip()
        self._sc.job_completed(id)

    def help_wait_for_jobs(self):
        print('''
        Does not returns until the jobs listed by their id are not completed
        > wait_for_jobs 42 23 56
        ''')

    @print_exception
    def do_wait_for_jobs(self, args):
        ids = re.split('[,\s]+', args.strip())
        ids = [int(x) for x in ids]
        self._sc.wait_for_jobs(ids)

    # classifiers

    def help_classifiers(self):
        print('''
        Lists classifiers
        ''')

    @print_exception
    def do_classifiers(self, args):
        classifiers = self._sc.classifiers()
        pprint(classifiers)

    def help_classifier_create(self):
        print('''
        Creates a classifier with a set of labels
        > classifier_create name label1 label2 label3
        the -o option at the end overwrites any existing classifier with the same name
        > classifier_create name label1 label2 label3 -o
        ''')

    @print_exception
    def do_classifier_create(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        args = args[1:]
        overwrite = False
        if args[-1] == '-o':
            overwrite = True
            args = args[:-1]
        labels = args
        self._sc.classifier_create(name, labels, overwrite)

    def help_classifier_delete(self):
        print('''
        Deletes a classifier
        > classifier_delete name
        ''')

    @print_exception
    def do_classifier_delete(self, args):
        classifier = args.strip()
        self._sc.classifier_delete(classifier)

    def help_classifier_duplicate(self):
        print('''
        Duplicates a classifier with a new name
        > classifier_create name new_name
        the -o option at the end overwrites any existing classifier with the same new_name
        > classifier_create name new_name -o
        ''')

    @print_exception
    def do_classifier_duplicate(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        new_name = args[1]
        overwrite = False
        if args[-1] == '-o':
            overwrite = True
        pprint(self._sc.classifier_duplicate(name, new_name, overwrite))

    def help_classifier_update(self):
        print('''
        Updates a classifier with an example
        > classifier_update classifier_name label_name text...
        ''')

    @print_exception
    def do_classifier_update(self, args):
        match = re.match('^([^,\s]+)[,\s]+([^,\s]+)[,\s]+(.+)$', args.strip())
        name = match.group(1)
        X = [match.group(3)]
        y = [match.group(2)]
        pprint(self._sc.classifier_update(name, X, y))

    def help_classifier_rename(self):
        print('''
        Renames a classifier with a new name
        > classifier_rename name new_name
        the -o option at the end overwrites any existing classifier with the same new_name
        > classifier_rename name new_name -o
        ''')

    @print_exception
    def do_classifier_rename(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        new_name = args[1]
        overwrite = False
        if args[-1] == '-o':
            overwrite = True
        self._sc.classifier_rename(name, new_name, overwrite)

    def help_classifier_labels(self):
        print('''
        Prints the labes for a classifier
        > classifier_labels classifiername
        ''')

    @print_exception
    def do_classifier_labels(self, args):
        labels = self._sc.classifier_labels(args.strip())
        pprint(labels)

    def help_classifier_rename_label(self):
        print('''
        Renames a label of a classifier with a new name
        > classifier_rename classifiername label_name new_label_name
        ''')

    @print_exception
    def do_classifier_rename_label(self, args):
        match = re.match('^([^,\s]+)[,\s]+([^,\s]+)[,\s]+(.+)$', args.strip())
        classifier_name = match.group(1)
        label_name = match.group(2)
        new_name = match.group(3)
        self._sc.classifier_rename_label(classifier_name, label_name, new_name)

    def help_classifier_download_training_data(self):
        print('''
        Downloads training data for a classifier to a file
        > classifier_download_training_data classifiername filename
        Add -o to overwrite the eventual already existing file
        ''')

    @print_exception
    def do_classifier_download_training_data(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        filename = args[1]
        overwrite = False
        if args[-1] == '-o':
            overwrite = True
        if not overwrite:
            if exists(filename):
                raise FileExistsError('File %s already exists' % filename)
        else:
            if exists(filename):
                remove(filename)
        with open(filename, mode='w', encoding='utf-8') as outfile:
            self._sc.classifier_download_training_data(name, outfile)

    def help_classifier_download_model(self):
        print('''
        Downloads the classification model for a classifier to a file
        > classifier_download_model classifiername filename
        Add -o to overwrite the eventual already existing file
        ''')

    @print_exception
    def do_classifier_download_model(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        filename = args[1]
        overwrite = False
        if args[-1] == '-o':
            overwrite = True
        if not overwrite:
            if exists(filename):
                raise FileExistsError('File %s already exists' % filename)
        else:
            if exists(filename):
                remove(filename)
        with open(filename, mode='w', encoding='utf-8') as outfile:
            self._sc.classifier_download_model(name, outfile)

    def help_classifier_upload_training_data(self):
        print('''
        Uploads training data to the classifiers defined in the file given as input
        > classifier_upload_training_data filename
        ''')

    @print_exception
    def do_classifier_upload_training_data(self, args):
        with open(args.strip(), mode='r', encoding='utf-8') as infile:
            pprint(self._sc.classifier_upload_training_data(infile))

    def help_classifier_upload_model(self):
        print('''
        Uploads a classification model to define a new classifier
        > classifier_upload_model classifiername filename
        adding -o eventually overwrites an already existing classifier with the same name
        ''')

    @print_exception
    def do_classifier_upload_model(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        filename = args[1]
        overwrite = False
        if args[-1] == '-o':
            overwrite = True
        with open(filename, mode='rb') as infile:
            pprint(self._sc.classifier_upload_model(name, infile, overwrite))

    def help_classifier_classify(self):
        print('''
        Classifies a piece of text
        > classifier_classify classifier_name text...
        assignedlabel
        >
        ''')

    @print_exception
    def do_classifier_classify(self, args):
        match = re.match('^([^,\s]+)[,\s]+(.+)$', args.strip())
        name = match.group(1)
        X = [match.group(2)]
        label = self._sc.classifier_classify(name, X)
        pprint(label)

    def help_classifier_score(self):
        print('''
        Classifies a piece of text
        > classifier_score classifier_name text...
        label1:score1 label2:score2 label3:score3
        ''')

    @print_exception
    def do_classifier_score(self, args):
        match = re.match('^([^,\s]+)[,\s]+(.+)$', args.strip())
        name = match.group(1)
        X = [match.group(2)]
        scores = self._sc.classifier_score(name, X)
        pprint(scores)

    def help_classifier_extract(self):
        print('''
        Extracts the given set of labels as distinct binary classifiers
        > classifier_extract classifier_name label1 label2...
        ''')

    @print_exception
    def do_classifier_extract(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        args = args[1:]
        labels = args
        pprint(self._sc.classifier_extract(name, labels))

    def help_classifier_combine(self):
        print('''
        Combines the labels of a set of classifiers into a new classifier
        > classifier_combine new_classifier classifier1 classifier2 classifier3...
        Any classifier that has only 'yes' and 'no' labels is considered itself as a label
        ''')

    @print_exception
    def do_classifier_combine(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        args = args[1:]
        sources = args
        pprint(self._sc.classifier_combine(name, sources))

    # datasets

    def help_datasets(self):
        print('''
        Lists datasets
        ''')

    @print_exception
    def do_datasets(self, args):
        datasets = self._sc.datasets()
        pprint(datasets)

    def help_dataset_create(self):
        print('''
        Creates a dataset
        > dataset_create datasetname
        ''')

    @print_exception
    def do_dataset_create(self, args):
        name = args.strip()
        self._sc.dataset_create(name)

    def help_dataset_add_document(self):
        print('''
        Add a document to a dataset
        > dataset_add_document datasetname documentname text...
        If the dataset does not exist, it is created
        ''')

    @print_exception
    def do_dataset_add_document(self, args):
        match = re.match('^([^,\s]+)[,\s]+([^,\s]+)[,\s]+(.+)$', args.strip())
        dataset_name = match.group(1)
        document_name = match.group(2)
        document_content = match.group(3)
        self._sc.dataset_add_document(dataset_name, document_name, document_content)

    def help_dataset_delete_document(self):
        print('''
        Deletes a document from a dataset
        > dataset_delete_document datasetname documentname
        ''')

    @print_exception
    def do_dataset_delete_document(self, args):
        args = re.split('[,\s]+', args.strip())
        dataset_name = args[0]
        document_name = args[1]
        self._sc.dataset_delete_document(dataset_name, document_name)

    def help_dataset_document_by_name(self):
        print('''
        Gets a document from a dataset, retrieving it by its unique name
        > dataset_document_by_name datasetname documentname
        ''')

    @print_exception
    def do_dataset_document_by_name(self, args):
        args = re.split('[,\s]+', args.strip())
        dataset_name = args[0]
        document_name = args[1]
        pprint(self._sc.datasets_document_by_name(dataset_name, document_name))

    def help_dataset_document_by_position(self):
        print('''
        Gets a document from a dataset, retrieving it by its order in insertion in the dataset
        > dataset_document_by_position datasetname position
        ''')

    @print_exception
    def do_dataset_document_by_position(self, args):
        args = re.split('[,\s]+', args.strip())
        dataset_name = args[0]
        position = int(args[1])
        pprint(self._sc.datasets_document_by_position(dataset_name, position))

    def help_dataset_rename(self):
        print('''
        Renames a dataset with a new name
        > dataset_rename name new_name
        No dataset with the new name must exist
        ''')

    @print_exception
    def do_dataset_rename(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        new_name = args[1]
        self._sc.dataset_rename(name, new_name)

    def help_dataset_delete(self):
        print('''
        Deletes a dataset
        > dataset_delete name
        ''')

    @print_exception
    def do_dataset_delete(self, args):
        dataset = args.strip()
        self._sc.dataset_delete(dataset)

    def help_dataset_upload(self):
        print('''
        Uploads document to a dataset from a file
        > dataset_upload datasetname filename
        ''')

    @print_exception
    def do_dataset_upload(self, args):
        match = re.match('^([^,\s]+)[,\s]+(.+)$', args.strip())
        datasetname = match.group(1)
        filename = match.group(2)
        with open(filename, mode='r', encoding='utf-8') as infile:
            pprint(self._sc.dataset_upload(datasetname, filename))

    def help_dataset_download(self):
        print('''
        Downloads a dataset to a file
        > dataset_download datsetname filename
        Add -o to overwrite the eventual already existing file
        ''')

    @print_exception
    def do_dataset_download(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        filename = args[1]
        overwrite = False
        if args[-1] == '-o':
            overwrite = True
        if not overwrite:
            if exists(filename):
                raise FileExistsError('File %s already exists' % filename)
        else:
            if exists(filename):
                remove(filename)
        with open(filename, mode='w', encoding='utf-8') as outfile:
            self._sc.dataset_download(name, outfile)

    def help_dataset_size(self):
        print('''
        Prints the size of a dataset
        > dataset_size classifiername
        ''')

    @print_exception
    def do_dataset_size(self, args):
        pprint(self._sc.dataset_size(args.strip()))

    def help_dataset_classify(self):
        print('''
        Automatically classifies a dataset using the listed classifiers
        > dataset_classify datasetname classifier1 classifier2...
        ''')

    @print_exception
    def do_dataset_classify(self, args):
        args = re.split('[,\s]+', args.strip())
        datasetname = args[0]
        classifiers = args[1:]
        pprint(self._sc.dataset_classify(datasetname, classifiers))

    def help_dataset_get_classification_jobs(self):
        print('''
        Prints the list of the available classifications for a dataset
        > dataset_get_classification_jobs datasetname
        ''')

    @print_exception
    def do_dataset_get_classification_jobs(self, args):
        name = args.strip()
        pprint(self._sc.dataset_get_classification_jobs(name))

    def help_dataset_download_classification(self):
        print('''
        Downloads a classification of a dataset to a file
        > dataset_download_classification datsetname filename
        Add -o to overwrite the eventual already existing file
        ''')

    @print_exception
    def do_dataset_download_classification(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        filename = args[1]
        overwrite = False
        if args[-1] == '-o':
            overwrite = True
        if not overwrite:
            if exists(filename):
                raise FileExistsError('File %s already exists' % filename)
        else:
            if exists(filename):
                remove(filename)
        with open(filename, mode='w', encoding='utf-8') as outfile:
            self._sc.dataset_download_classification(name, outfile)

    def help_dataset_delete_classification(self):
        print('''
        Delete a classification of a dataset
        ''')

    @print_exception
    def do_dataset_delete_classification(self, args):
        id = args.strip()
        self._sc.dataset_delete_classification(id)


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
