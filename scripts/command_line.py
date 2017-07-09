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
        username = args.strip()
        if len(username) == 0:
            username = input('Username: ').strip()
        password = getpass.getpass('Password: ')
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
        args = args[1:]
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
        > classifier_delete name
        '''
        classifier = args.strip()
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
        pprint(self._sc.classifier_duplicate(name, new_name, overwrite))

    @print_exception
    def do_classifier_update(self, args):
        '''
        Updates a classifier with an example
        > classifier_update classifier_name label_name text...
        '''
        match = re.match('^([^,\s]+)[,\s]+([^,\s]+)[,\s]+(.+)$', args.strip())
        name = match.group(1)
        X = [match.group(3)]
        y = [match.group(2)]
        pprint(self._sc.classifier_update(name, X, y))

    @print_exception
    def do_classifier_rename(self, args):
        '''
        Renames a classifier with a new name
        > classifier_rename name new_name
        the -o option at the end overwrites any existing classifier with the same new_name
        > classifier_rename name new_name -o
        '''
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        new_name = args[1]
        overwrite = False
        if args[-1] == '-o':
            overwrite = True
        self._sc.classifier_rename(name, new_name, overwrite)

    @print_exception
    def do_classifier_labels(self, args):
        '''
        Prints the labes for a classifier
        > classifier_labels classifiername
        '''
        labels = self._sc.classifier_labels(args.strip())
        pprint(labels)

    @print_exception
    def do_classifier_rename_label(self, args):
        '''
        Renames a label of a classifier with a new name
        > classifier_rename classifiername label_name new_label_name
        '''
        match = re.match('^([^,\s]+)[,\s]+([^,\s]+)[,\s]+(.+)$', args.strip())
        classifier_name = match.group(1)
        label_name = match.group(2)
        new_name = match.group(3)
        self._sc.classifier_rename_label(classifier_name, label_name, new_name)

    @print_exception
    def do_classifier_download_training_data(self, args):
        '''
        Downloads training data for a classifier to a file
        > classifier_download_training_data classifiername filename
        Add -o to overwrite the eventual already existing file
        '''
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

    @print_exception
    def do_classifier_upload_training_data(self, args):
        '''
        Uploads training data to the classifiers defined in the file given as input
        > classifier_upload_training_data filename
        '''
        with open(args, mode='r', encoding='utf-8') as infile:
            pprint(self._sc.classifier_upload_training_data(infile))

    @print_exception
    def do_classifier_classify(self, args):
        '''
        Classifies a piece of text
        > classifier_classify classifier_name text...
        assignedlabel
        >
        '''
        match = re.match('^([^,\s]+)[,\s]+(.+)$', args.strip())
        name = match.group(1)
        X = [match.group(2)]
        label = self._sc.classifier_classify(name, X)
        pprint(label)

    @print_exception
    def do_classifier_score(self, args):
        '''
        Classifies a piece of text
        > classifier_score classifier_name text...
        label1:score1 label2:score2 label3:score3
        >
        '''
        match = re.match('^([^,\s]+)[,\s]+(.+)$', args.strip())
        name = match.group(1)
        X = [match.group(2)]
        scores = self._sc.classifier_score(name, X)
        pprint(scores)

    @print_exception
    def do_classifier_extract(self, args):
        '''
        Extracts the given set of labels as distinct binary classifiers
        > classifier_extract classifier_name label1 label2...
        '''
        args = re.split('[,\s]+', args)
        name = args[0]
        args = args[1:]
        labels = args
        pprint(self._sc.classifier_extract(name, labels))

    @print_exception
    def do_classifier_combine(self, args):
        '''
        Combines the labels of a set of classifiers into a new classifier
        > classifier_combine new_classifier classifier1 classifier2 classifier3...
        Any classifier that has only 'yes' and 'no' labels is considered itself as a label
        '''
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        args = args[1:]
        sources = args
        pprint(self._sc.classifier_combine(name, sources))

    # datasets

    @print_exception
    def do_datasets(self, args):
        '''
        Lists datasets
        '''
        datasets = self._sc.datasets()
        pprint(datasets)

    @print_exception
    def do_dataset_create(self, args):
        '''
        Creates a dataset
        > dataset_create datasetname
        '''
        name = args.strip()
        self._sc.dataset_create(name)

    @print_exception
    def do_dataset_add_document(self, args):
        '''
        Add a document to a dataset
        > dataset_add_document datasetname documentname text...
        If the dataset does not exist, it is created
        '''
        match = re.match('^([^,\s]+)[,\s]+([^,\s]+)[,\s]+(.+)$', args.strip())
        dataset_name = match.group(1)
        document_name = match.group(2)
        document_content = match.group(3)
        pprint(self._sc.dataset_add_document(dataset_name, document_name, document_content))

    @print_exception
    def do_dataset_delete_document(self, args):
        '''
        Deletes a document from a dataset
        > dataset_delete_document datasetname documentname
        '''
        args = re.split('[,\s]+', args)
        dataset_name = args[0]
        document_name = args[1]
        self._sc.dataset_delete_document(dataset_name,document_name)

    @print_exception
    def do_dataset_rename(self, args):
        '''
        Renames a dataset with a new name
        > dataset_rename name new_name
        No dataset with the new name must exist
        '''
        args = re.split('[,\s]+', args)
        name = args[0]
        new_name = args[1]
        self._sc.dataset_rename(name, new_name)

    @print_exception
    def do_dataset_delete(self, args):
        '''
        Deletes a dataset
        > dataset_delete name
        '''
        dataset = args.strip()
        self._sc.classifier_delete(dataset)

    @print_exception
    def do_dataset_upload(self, args):
        '''
        Uploads document to a dataset from a file
        > dataset_upload datasetname filename
        '''
        match = re.match('^([^,\s]+)[,\s]+(.+)$', args.strip())
        datasetname = match.group(1)
        filename = match.group(2)
        with open(filename, mode='r', encoding='utf-8') as infile:
            pprint(self._sc.dataset_upload(datasetname,filename))

    @print_exception
    def do_dataset_download(self, args):
        '''
        Downloads a dataset to a file
        > dataset_download datsetname filename
        Add -o to overwrite the eventual already existing file
        '''
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

    @print_exception
    def do_dataset_size(self, args):
        '''
        Prints the size of a dataset
        > dataset_size classifiername
        '''
        pprint(self._sc.dataset_size(args.strip()))

    @print_exception
    def do_dataset_classify(self, args):
        '''
        Automatically classifies a dataset using the listed classifiers
        > dataset_classify datasetname classifier1 classifier2...
        '''
        args = re.split('[,\s]+', args.strip())
        datasetname = args[0]
        classifiers = args[1:]
        pprint(self._sc.dataset_classify(datasetname,classifiers))

    @print_exception
    def do_dataset_get_classification_jobs(self, args):
        '''
        Prints the list of the available classifications for a dataset
        > dataset_get_classification_jobs datasetname
        '''
        name = args.strip()
        pprint(self._sc.dataset_get_classification_jobs(name))

    @print_exception
    def do_dataset_download_classification(self, args):
        '''
        Downloads a classification of a dataset to a file
        > dataset_download_classification datsetname filename
        Add -o to overwrite the eventual already existing file
        '''
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

    @print_exception
    def do_dataset_delete_classification(self, args):
        '''
        Delete a classification of a dataset
        '''
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
