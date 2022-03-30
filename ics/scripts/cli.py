import getpass
import re
import sys
from argparse import ArgumentParser
from cmd import Cmd
from os import remove
from os.path import exists
from pprint import pprint

from ics.client import ClientSession

__author__ = 'Andrea Esuli'


def print_exception(fn):
    def wrapped(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            print('Error: ' + str(e))

    return wrapped


class CommandLine(Cmd):
    def __init__(self, protocol, host, port, classifier_path, dataset_path,
                 jobs_path, user_auth_path, key_auth_path, ip_auth_path):
        self._sc = ClientSession(protocol, host, port, classifier_path, dataset_path, jobs_path, user_auth_path,
                                 key_auth_path, ip_auth_path)
        super().__init__()
        self.prompt = '> '

    def emptyline(self):
        pass

    def help_exit(self):
        print('Exit the program')

    def do_exit(self, args):
        sys.exit()

    # users

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
        pprint(self._sc.login(username, getpass.getpass('Password: ')))

    def help_whoami(self):
        print('''
        Returns the name of the currently logged in user, or an error if not logged in.
        ''')

    @print_exception
    def do_whoami(self, args):
        pprint(self._sc.whoami())

    def help_logout(self):
        print('''
        Logout command.
        This command does not terminate the command prompt.
        ''')

    @print_exception
    def do_logout(self, args):
        pprint(self._sc.logout())

    def help_user_info(self):
        print('''
        Prints the list of registered usernames
        ''')

    @print_exception
    def do_user_info(self, args):
        users = self._sc.user_info()
        pprint(users)

    def help_user_count(self):
        print('''
        Number of registered users
        ''')

    @print_exception
    def do_user_count(self, args):
        count = self._sc.user_count()
        pprint(count)

    def help_user_create(self):
        print('''
        Creates a new user, given a name, a hourly limit, and a total request limit (use -1 for no limit)
        > user_create username 100 10000
        Password: <typing of password is hidden>
        ''')

    @print_exception
    def do_user_create(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        hourly_limit = args[1]
        request_limit = args[2]
        if len(name) == 0:
            name = input('Username: ').strip()
        password = getpass.getpass('Password:')
        confirmpassword = getpass.getpass('Confirm password:')
        if password == confirmpassword:
            pprint(self._sc.user_create(name, password, hourly_limit, request_limit))
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
        confirmpassword = getpass.getpass('Confirm password:')
        if password == confirmpassword:
            pprint(self._sc.user_change_password(username, password))
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
        pprint(self._sc.user_delete(username))

    def help_user_set_hourly_limit(self):
        print('''
        Sets the hourly request limit for a user
        > user_set_hourly_limit username 100
        ''')

    @print_exception
    def do_user_set_hourly_limit(self, args):
        args = re.split('[,\s]+', args.strip())
        username = args[0]
        hourly_limit = args[1]
        pprint(self._sc.user_set_hourly_limit(username, hourly_limit))

    def help_user_set_request_limit(self):
        print('''
        Sets the total request limit for a user
        > user_set_request_limit username 10000
        ''')

    @print_exception
    def do_user_set_request_limit(self, args):
        args = re.split('[,\s]+', args.strip())
        username = args[0]
        request = args[1]
        pprint(self._sc.user_set_request_limit(username, request))

    def help_user_set_current_request_counter(self):
        print('''
        Sets the current number of request for a user
        > user_set_current_request_counter username 0
        ''')

    @print_exception
    def do_user_set_current_request_counter(self, args):
        args = re.split('[,\s]+', args.strip())
        username = args[0]
        count = args[1]
        pprint(self._sc.user_set_current_request_counter(username, count))

    def help_user_version(self):
        print('''
        Prints the version of the user authentication web service
        ''')

    @print_exception
    def do_user_version(self, args):
        pprint(self._sc.user_version())

    # keys

    def help_key_info(self):
        print('''
        Prints the list of registered keys
        ''')

    @print_exception
    def do_key_info(self, args):
        keys = self._sc.key_info()
        pprint(keys)

    def help_key_count(self):
        print('''
        Number of registered keys
        ''')

    @print_exception
    def do_key_count(self, args):
        count = self._sc.key_count()
        pprint(count)

    def help_key_create(self):
        print('''
        Creates a new key, given a name, a hourly limit, and a total request limit (use -1 for no limit)
        > key_create name 100 10000
        Key: 12345678...
        ''')

    @print_exception
    def do_key_create(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        hourly_limit = args[1]
        request_limit = args[2]
        pprint(self._sc.key_create(name, hourly_limit, request_limit))

    def help_key_use(self):
        print('''
        Registers a key to be used with the successive requests
        > key_use 12345678...
        ''')

    @print_exception
    def do_key_use(self, args):
        pprint(self._sc.key_use(args.strip()))

    def help_key_disable_use(self):
        print('''
        Removes any registered key
        > key_disable_use
        ''')

    def do_key_disable_use(self, args):
        pprint(self._sc.key_disable_use())

    def help_key_delete(self):
        print('''
        Deletes a key
        > key_delete 12345678...
        ''')

    @print_exception
    def do_key_delete(self, args):
        key = args.strip()
        pprint(self._sc.key_delete(key))

    def help_key_set_hourly_limit(self):
        print('''
        Sets the hourly request limit for a key
        > key_set_hourly_limit 12345678... 100
        ''')

    @print_exception
    def do_key_set_hourly_limit(self, args):
        args = re.split('[,\s]+', args.strip())
        key = args[0]
        hourly_limit = args[1]
        pprint(self._sc.key_set_hourly_limit(key, hourly_limit))

    def help_key_set_request_limit(self):
        print('''
        Sets the total request limit for a key
        > key_set_request_limit 12345678... 10000
        ''')

    @print_exception
    def do_key_set_request_limit(self, args):
        args = re.split('[,\s]+', args.strip())
        key = args[0]
        request = args[1]
        pprint(self._sc.key_set_request_limit(key, request))

    def help_key_set_current_request_counter(self):
        print('''
        Sets the current number of request for a key
        > key_set_current_request_counter 12345678... 0
        ''')

    @print_exception
    def do_key_set_current_request_counter(self, args):
        args = re.split('[,\s]+', args.strip())
        key = args[0]
        count = args[1]
        pprint(self._sc.key_set_current_request_counter(key, count))

    def help_key_version(self):
        print('''
        Prints the version of the key authentication web service
        ''')

    @print_exception
    def do_key_version(self, args):
        pprint(self._sc.key_version())

    # ips

    def help_ip_info(self):
        print('''
        Prints the list of registered ips
        ''')

    @print_exception
    def do_ip_info(self, args):
        ips = self._sc.ip_info()
        pprint(ips)

    def help_ip_count(self):
        print('''
        Number of registered ips
        ''')

    @print_exception
    def do_ip_count(self, args):
        count = self._sc.ip_count()
        pprint(count)

    def help_ip_create(self):
        print('''
        Creates a new ip, given a ip, a hourly limit, and a total request limit (use -1 for no limit)
        > ip_create ip 100 10000
        ''')

    @print_exception
    def do_ip_create(self, args):
        args = re.split('[,\s]+', args.strip())
        ip = args[0]
        hourly_limit = args[1]
        request_limit = args[2]
        pprint(self._sc.ip_create(ip, hourly_limit, request_limit))

    def help_ip_delete(self):
        print('''
        Deletes an ip
        > ip_delete ip
        ''')

    @print_exception
    def do_ip_delete(self, args):
        ip = args.strip()
        pprint(self._sc.ip_delete(ip))

    def help_ip_set_hourly_limit(self):
        print('''
        Sets the hourly request limit for an ip
        > ip_set_hourly_limit ip 100
        ''')

    @print_exception
    def do_ip_set_hourly_limit(self, args):
        args = re.split('[,\s]+', args.strip())
        ip = args[0]
        hourly_limit = args[1]
        pprint(self._sc.ip_set_hourly_limit(ip, hourly_limit))

    def help_ip_set_request_limit(self):
        print('''
        Sets the total request limit for an ip
        > ip_set_request_limit ip 10000
        ''')

    @print_exception
    def do_ip_set_request_limit(self, args):
        args = re.split('[,\s]+', args.strip())
        ip = args[0]
        request = args[1]
        pprint(self._sc.ip_set_request_limit(ip, request))

    def help_ip_set_current_request_counter(self):
        print('''
        Sets the current number of request for an ip
        > ip_set_current_request_counter ip 0
        ''')

    @print_exception
    def do_ip_set_current_request_counter(self, args):
        args = re.split('[,\s]+', args.strip())
        ip = args[0]
        count = args[1]
        pprint(self._sc.ip_set_current_request_counter(ip, count))

    def help_ip_version(self):
        print('''
        Prints the version of the ip authentication web service
        ''')

    @print_exception
    def do_ip_version(self, args):
        pprint(self._sc.ip_version())

    # jobs

    def help_job_info(self):
        print('''
        Lists jobs
        ''')

    @print_exception
    def do_job_info(self, args):
        jobs = self._sc.job_info()
        pprint(jobs)

    def help_job_count(self):
        print('''
        Number of jobs
        ''')

    @print_exception
    def do_job_count(self, args):
        count = self._sc.job_count()
        pprint(count)

    def help_job_delete(self):
        print('''
        Deletes a job by its id
        > job_delete 42
        ''')

    @print_exception
    def do_job_delete(self, args):
        id = args.strip()
        pprint(self._sc.job_delete(id))

    def help_job_rerun(self):
        print('''
        Reruns a job by its id
        > job_rerun 42
        ''')

    @print_exception
    def do_job_rerun(self, args):
        id = args.strip()
        pprint(self._sc.job_rerun(id))

    def help_job_delete_all(self):
        print('''
        Deletes all jobs
        ''')

    @print_exception
    def do_job_delete_all(self, args):
        pprint(self._sc.job_delete_all())

    def help_job_delete_all_done(self):
        print('''
        Deletes all completed jobs
        ''')

    @print_exception
    def do_job_delete_all_done(self, args):
        pprint(self._sc.job_delete_all_done())

    def help_job_delete_all_errors(self):
        print('''
        Deletes all jobs ended with an error
        ''')

    @print_exception
    def do_job_delete_all_errors(self, args):
        pprint(self._sc.job_delete_all_errors())

    def help_job_delete_all_not_running(self):
        print('''
        Deletes all jobs not currently running
        ''')

    @print_exception
    def do_job_delete_all_not_running(self, args):
        pprint(self._sc.job_delete_all_not_running())

    def help_job_completed(self):
        print('''
        Checks if a job is completed or not
        > job_completed 42
        True
        ''')

    @print_exception
    def do_job_completed(self, args):
        id = args.strip()
        pprint(self._sc.job_completed(id))

    def help_job_wait(self):
        print('''
        Does not return until the jobs listed by their id are not completed
        > wait_for_jobs 42 23 56
        ''')

    @print_exception
    def do_job_wait(self, args):
        ids = re.split('[,\s]+', args.strip())
        ids = [int(x) for x in ids]
        pprint(self._sc.job_wait(ids))

    def help_job_lock_info(self):
        print('''
        Lists locks
        ''')

    @print_exception
    def do_job_lock_info(self, args):
        locks = self._sc.job_lock_info()
        pprint(locks)

    def help_job_lock_count(self):
        print('''
        Number of locks
        ''')

    @print_exception
    def do_job_lock_count(self, args):
        count = self._sc.job_lock_count()
        pprint(count)

    def help_job_lock_delete(self):
        print('''
        Deletes a lock by its id
        > job_lock_delete 42
        ''')

    @print_exception
    def do_job_lock_delete(self, args):
        id = args.strip()
        pprint(self._sc.job_lock_delete(id))

    def help_job_version(self):
        print('''
        Prints the version of the job web service
        ''')

    @print_exception
    def do_job_version(self, args):
        pprint(self._sc.job_version())

    # classifiers

    def help_classifier_info(self):
        print('''
        Lists classifiers
        ''')

    @print_exception
    def do_classifier_info(self, args):
        classifiers = self._sc.classifier_info()
        pprint(classifiers)

    def help_classifier_count(self):
        print('''
        Number of classifiers
        ''')

    @print_exception
    def do_classifier_count(self, args):
        count = self._sc.classifier_count()
        pprint(count)

    def help_classifier_types(self):
        print('''
        Lists the types of classifiers
        ''')

    @print_exception
    def do_classifier_types(self, args):
        types = self._sc.classifier_types()
        pprint(types)

    def help_classifier_create(self):
        print('''
        Creates a classifier with a set of labels
        > classifier_create name type label1 label2 label3
        the -o option at the end overwrites any existing classifier with the same name
        > classifier_create name type label1 label2 label3 -o
        ''')

    @print_exception
    def do_classifier_create(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        type = args[1]
        args = args[2:]
        overwrite = False
        if args[-1] == '-o':
            overwrite = True
            args = args[:-1]
        labels = args
        pprint(self._sc.classifier_create(name, labels, type, overwrite))

    def help_classifier_delete(self):
        print('''
        Deletes a classifier
        > classifier_delete name
        ''')

    @print_exception
    def do_classifier_delete(self, args):
        classifier = args.strip()
        pprint(self._sc.classifier_delete(classifier))

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

    def help_classifier_description(self):
        print('''
        Sets a description for a classifier
        > classifier_description classifier_name description
        ''')

    @print_exception
    def do_classifier_description(self, args):
        match = re.match('^([^,\s]+)[,\s]+(.+)$', args.strip())
        name = match.group(1)
        description = match.group(2)
        pprint(self._sc.classifier_set_description(name, description))

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
        pprint(self._sc.classifier_rename(name, new_name, overwrite))

    def help_classifier_set_public(self):
        print('''
        Makes classifier public or not public
        > classifier_set_public name [true|false]
        ''')

    @print_exception
    def do_classifier_set_public(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        if args[1] in ['true', 'True', 't', 'T']:
            public = True
        elif args[1] in ['false', 'False', 'f', 'F']:
            public = False
        else:
            pprint(f'Flag not in [true,false]: {args[1]}')
            return
        pprint(self._sc.classifier_set_public(name, public))

    def help_classifier_label_info(self):
        print('''
        Prints the labes for a classifier
        > classifier_label_info classifiername
        ''')

    @print_exception
    def do_classifier_label_info(self, args):
        labels = self._sc.classifier_label_info(args.strip())
        pprint(labels)

    def help_classifier_label_rename(self):
        print('''
        Renames a label of a classifier with a new name
        > classifier_label_rename classifiername label_name new_label_name
        ''')

    @print_exception
    def do_classifier_label_rename(self, args):
        match = re.match('^([^,\s]+)[,\s]+([^,\s]+)[,\s]+(.+)$', args.strip())
        classifier_name = match.group(1)
        label_name = match.group(2)
        new_name = match.group(3)
        pprint(self._sc.classifier_label_rename(classifier_name, label_name, new_name))


    def help_classifier_label_add(self):
        print('''
        Add a label to a classifier 
        > classifier_label_add classifiername label_name
        ''')

    @print_exception
    def do_classifier_label_add(self, args):
        match = re.match('^([^,\s]+)[,\s]+(.+)$', args.strip())
        classifier_name = match.group(1)
        label_name = match.group(2)
        pprint(self._sc.classifier_label_add(classifier_name, label_name))


    def help_classifier_label_delete(self):
        print('''
        Deletes a label from a classifier
        > classifier_label_delete classifiername label_name
        ''')

    @print_exception
    def do_classifier_label_delete(self, args):
        match = re.match('^([^,\s]+)[,\s]+(.+)$', args.strip())
        classifier_name = match.group(1)
        label_name = match.group(2)
        pprint(self._sc.classifier_label_delete(classifier_name, label_name))

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
            pprint(self._sc.classifier_download_training_data(name, outfile))

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
        with open(filename, mode='wb') as outfile:
            pprint(self._sc.classifier_download_model(name, outfile))

    def help_classifier_upload_training_data(self):
        print('''
        Uploads training data to the classifiers defined in the file given as input
        > classifier_upload_training_data filename type
        ''')

    @print_exception
    def do_classifier_upload_training_data(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        type = args[1]
        with open(name, mode='r', encoding='utf-8') as infile:
            pprint(self._sc.classifier_upload_training_data(infile, type))

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
        > classifier_extract classifier_name type label1 label2...
        ''')

    @print_exception
    def do_classifier_extract(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        type = args[1]
        args = args[2:]
        labels = args
        pprint(self._sc.classifier_extract(name, type, labels))

    def help_classifier_merge(self):
        print('''
        Merges the labels of a set of classifiers into a new classifier
        > classifier_merge new_classifier type classifier1 classifier2 classifier3...
        ''')

    @print_exception
    def do_classifier_merge(self, args):
        args = re.split('[,\s]+', args.strip())
        name = args[0]
        type = args[1]
        args = args[2:]
        sources = args
        pprint(self._sc.classifier_merge(name, sources, type))

    def help_classifier_version(self):
        print('''
        Prints the version of the classifier web service
        ''')

    @print_exception
    def do_classifier_version(self, args):
        pprint(self._sc.classifier_version())

    # datasets

    def help_dataset_info(self):
        print('''
        Lists datasets
        ''')

    @print_exception
    def do_dataset_info(self, args):
        datasets = self._sc.dataset_info()
        pprint(datasets)

    def help_dataset_create(self):
        print('''
        Creates a dataset
        > dataset_create datasetname
        ''')

    def help_dataset_count(self):
        print('''
        Number of datasets
        ''')

    @print_exception
    def do_dataset_count(self, args):
        count = self._sc.dataset_count()
        pprint(count)

    @print_exception
    def do_dataset_create(self, args):
        name = args.strip()
        if len(name) == 0:
            print('Must specify a dataset name')
        pprint(self._sc.dataset_create(name))

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
        pprint(self._sc.dataset_add_document(dataset_name, document_name, document_content))

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
        pprint(self._sc.dataset_delete_document(dataset_name, document_name))

    def help_dataset_description(self):
        print('''
        Sets a description for a dataset
        > dataset_description dataset_name description
        ''')

    @print_exception
    def do_dataset_description(self, args):
        match = re.match('^([^,\s]+)[,\s]+(.+)$', args.strip())
        name = match.group(1)
        description = match.group(2)
        pprint(self._sc.dataset_set_description(name, description))

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
        pprint(self._sc.dataset_rename(name, new_name))

    def help_dataset_delete(self):
        print('''
        Deletes a dataset
        > dataset_delete name
        ''')

    @print_exception
    def do_dataset_delete(self, args):
        dataset = args.strip()
        pprint(self._sc.dataset_delete(dataset))

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
            pprint(self._sc.dataset_upload(datasetname, infile))

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
            pprint(self._sc.dataset_download(name, outfile))

    def help_dataset_size(self):
        print('''
        Prints the size of a dataset
        > dataset_size classifiername
        ''')

    @print_exception
    def do_dataset_size(self, args):
        pprint(self._sc.dataset_size(args.strip()))

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
        pprint(self._sc.dataset_document_by_name(dataset_name, document_name))

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
        pprint(self._sc.dataset_document_by_position(dataset_name, position))

    def help_dataset_most_uncertain_document_id(self):
        print('''
        Prints the name of the document that the classifiers classifies with the highest uncertainty
        > dataset_most_uncertain_document_id dataset_name classifier_name
        ''')

    def do_dataset_most_uncertain_document_id(self, args):
        args = re.split('[,\s]+', args.strip())
        dataset_name = args[0]
        classifier_name = args[1]
        pprint(self._sc.dataset_most_uncertain_document_id(dataset_name, classifier_name))

    def help_dataset_most_certain_document_id(self):
        print('''
        Prints the name of the document that the classifiers classifies with the highest confidence
        > dataset_most_certain_document_id dataset_name classifier_name
        ''')

    def do_dataset_most_certain_document_id(self, args):
        args = re.split('[,\s]+', args.strip())
        dataset_name = args[0]
        classifier_name = args[1]
        pprint(self._sc.dataset_most_certain_document_id(dataset_name, classifier_name))

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

    def help_dataset_classification_info(self):
        print('''
        Prints the list of the available classifications for a dataset
        > dataset_classification_info datasetname
        ''')

    @print_exception
    def do_dataset_classification_info(self, args):
        name = args.strip()
        pprint(self._sc.dataset_classification_info(name))

    def help_dataset_classification_download(self):
        print('''
        Downloads a classification of a dataset to a file given the id of the classification
        > dataset_classification_download id filename
        Add -o to overwrite the eventual already existing file
        ''')

    @print_exception
    def do_dataset_classification_download(self, args):
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
            pprint(self._sc.dataset_classification_download(name, outfile))

    def help_dataset_classification_delete(self):
        print('''
        Delete a classification of a dataset
        > dataset_classification_delete id
        ''')

    @print_exception
    def do_dataset_classification_delete(self, args):
        id = args.strip()
        pprint(self._sc.dataset_classification_delete(id))

    def help_dataset_version(self):
        print('''
        Prints the version of the dataset web service
        ''')

    @print_exception
    def do_dataset_version(self, args):
        pprint(self._sc.dataset_version())

    def help_version(self):
        print('''
        Prints the version of all the services''')

    def do_version(self, args):
        print('client: ', self._sc.version())
        print('user auth service: ', self._sc.user_version())
        print('key auth service: ', self._sc.key_version())
        print('ip auth service: ', self._sc.ip_version())
        print('job service: ', self._sc.job_version())
        print('classifier service: ', self._sc.classifier_version())
        print('dataset service: ', self._sc.dataset_version())


def main():
    parser = ArgumentParser()
    parser.add_argument('--protocol', help='host protocol', type=str, default='http')
    parser.add_argument('--host', help='host server address', type=str, default='127.0.0.1')
    parser.add_argument('--port', help='host server port', type=int, default=8080)
    parser.add_argument('--user_auth_path', help='server path of the user auth web service', type=str,
                        default='/service/userauth')
    parser.add_argument('--key_auth_path', help='server path of the key auth web service', type=str,
                        default='/service/keyauth')
    parser.add_argument('--ip_auth_path', help='server path of the ip auth web service', type=str,
                        default='/service/ipauth')
    parser.add_argument('--classifier_path', help='server path of the classifier web service', type=str,
                        default='/service/classifiers')
    parser.add_argument('--dataset_path', help='server path of the dataset web service', type=str,
                        default='/service/datasets')
    parser.add_argument('--jobs_path', help='server path of the jobs web service', type=str,
                        default='/service/jobs')
    args = parser.parse_args(sys.argv[1:])

    command_line = CommandLine(args.protocol, args.host, args.port, args.classifier_path, args.dataset_path,
                               args.jobs_path, args.user_auth_path, args.key_auth_path, args.ip_auth_path)

    command_line.cmdloop('Welcome, type help to have a list of commands')


if __name__ == "__main__":
    main()
