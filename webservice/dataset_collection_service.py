import csv
import os
import random
import shutil
from uuid import uuid4

import cherrypy
import numpy as np
from cherrypy.lib.static import serve_file

from db.sqlalchemydb import SQLAlchemyDB, Job, Label
from util.util import get_fully_portable_file_name

__author__ = 'Andrea Esuli'

MAX_BATCH_SIZE = 1000
CSV_LARGE_FIELD = 1024 * 1024 * 10

QUICK_CLASSIFICATION_BATCH_SIZE = 100


class DatasetCollectionService(object):
    def __init__(self, db_connection_string, data_dir):
        self._db_connection_string = db_connection_string
        self._db = SQLAlchemyDB(db_connection_string)
        self._download_dir = os.path.join(data_dir, 'downloads')
        self._upload_dir = os.path.join(data_dir, 'uploads')

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def info(self, page=None, page_size=50):
        result = []
        if page is None:
            names = self._db.dataset_names()
        else:
            names = self._db.dataset_names()[int(page) * int(page_size):(int(page) + 1) * int(page_size)]
        for name in names:
            dataset_info = dict()
            dataset_info['name'] = name
            dataset_info['created'] = str(self._db.get_dataset_creation_time(name))
            dataset_info['updated'] = str(self._db.get_dataset_last_update_time(name))
            dataset_info['size'] = self._db.get_dataset_size(name)
            result.append(dataset_info)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def count(self):
        return str(len(list(self._db.dataset_names())))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def create(self, name):
        name = name.strip()
        if len(name) == 0:
            cherrypy.response.status = 400
            return 'Must specify a dataset name'
        self._db.create_dataset(name)
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def add_document(self, name, document_name, document_content):
        if not self._db.dataset_exists(name):
            self._db.create_dataset(name)
        self._db.create_dataset_documents(name, (document_name, document_content))
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def delete_document(self, name, document_name):
        if not self._db.dataset_exists(name):
            cherrypy.response.status = 404
            return '%s does not exits' % name
        self._db.delete_dataset_document(name, document_name)
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def upload(self, **data):
        try:
            dataset_name = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a name'
        try:
            file = data['file']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must upload a file'

        if not self._db.dataset_exists(dataset_name):
            self._db.create_dataset(dataset_name)

        filename = 'dataset %s %s.csv' % (dataset_name, uuid4())
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(self._upload_dir, filename)
        with open(fullpath, 'wb') as outfile:
            shutil.copyfileobj(file.file, outfile)

        job_id = self._db.create_job(_create_dataset_documents, (self._db_connection_string, dataset_name, fullpath),
                                     description='upload to dataset \'%s\'' % dataset_name)

        return [job_id]

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def rename(self, name, new_name):
        try:
            self._db.rename_dataset(name, new_name)
        except KeyError:
            cherrypy.response.status = 404
            return '%s does not exits' % name
        except Exception as e:
            cherrypy.response.status = 500
            return 'Error (%s)' % str(e)
        else:
            return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def delete(self, name):
        job_id = self._db.create_job(_delete_dataset,
                                     (self._db_connection_string, name),
                                     description='delete dataset \'%s\'' % name)
        return [job_id]

    @cherrypy.expose
    def download(self, name):
        if not self._db.dataset_exists(name):
            cherrypy.response.status = 404
            return '\'%s\' does not exits' % name
        filename = 'dataset %s %s.csv' % (name, str(self._db.get_dataset_last_update_time(name)))
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(self._download_dir, filename)
        if not os.path.isfile(fullpath):
            try:
                with open(fullpath, 'w', encoding='utf-8', newline='') as file:
                    writer = csv.writer(file, lineterminator='\n')
                    for document in self._db.get_dataset_documents_by_name(name):
                        writer.writerow([document.external_id, document.text])
            except:
                os.unlink(fullpath)

        return serve_file(fullpath, "text/csv", "attachment")

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def size(self, name):
        if not self._db.dataset_exists(name):
            cherrypy.response.status = 404
            return '\'%s\' does not exits' % name
        return str(self._db.get_dataset_size(name))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def document_by_name(self, name, document_name):
        if not self._db.dataset_exists(name):
            cherrypy.response.status = 404
            return '\'%s\' does not exits' % name
        document = self._db.get_dataset_document_by_name(name, document_name)
        if document is not None:
            result = dict()
            result['external_id'] = document.external_id
            result['text'] = document.text
            result['created'] = str(document.creation)
            return result
        else:
            cherrypy.response.status = 404
            return 'Document with name \'%i\' does not exits in \'%s\'' % (document_name, name)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def document_by_position(self, name, position):
        position = int(position)
        if not self._db.dataset_exists(name):
            cherrypy.response.status = 404
            return '\'%s\' does not exits' % name
        document = self._db.get_dataset_document_by_position(name, position)
        if document is not None:
            result = dict()
            result['external_id'] = document.external_id
            result['text'] = document.text
            result['created'] = str(document.creation)
            return result
        else:
            cherrypy.response.status = 404
            return 'Position %i does not exits in \'%s\'' % (position, name)

    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def documents_without_labels_count(self, dataset_name, classifier_name):
        return str(self._db.get_dataset_documents_without_labels_count(dataset_name, classifier_name))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def most_uncertain_document_id(self, name, classifier_name):
        dataset_size = self._db.get_dataset_size(name)
        offset = random.randint(0, max(0, dataset_size - QUICK_CLASSIFICATION_BATCH_SIZE))
        X = list()
        doc_ids = list()
        for doc in self._db.get_dataset_documents_without_labels(name, classifier_name, offset,
                                                                 QUICK_CLASSIFICATION_BATCH_SIZE):
            X.append(doc.text)
            doc_ids.append(doc.id)

        if len(X) == 0:
            cherrypy.response.status = 400
            return f'No unlabeled documents in dataset \'{name}\' for classifier \'{classifier_name}\''

        scores = self._db.score(classifier_name, X)
        positions_scores = list()
        for i, dict_ in enumerate(scores):
            probs = self._softmax(list(dict_.values()))
            probs.sort()
            diff = probs[-1] - probs[-2]
            positions_scores.append((i, diff))
        positions_scores.sort(key=lambda x: x[1])
        return self._db.get_dataset_document_position_by_id(name, doc_ids[positions_scores[0][0]])

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def random_unlabeled_document_id(self, name, classifier_name):
        dataset_size = self._db.get_dataset_size(name)
        offset = random.randint(0, max(0, dataset_size - QUICK_CLASSIFICATION_BATCH_SIZE))
        try:
            doc_id = self._db.get_dataset_documents_without_labels(name, classifier_name, offset, 1)[0].id
            return self._db.get_dataset_document_position_by_id(name, doc_id)
        except:
            cherrypy.response.status = 400
            return f'No unlabeled documents in dataset \'{name}\' for classifier \'{classifier_name}\''

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def most_certain_document_id(self, name, classifier_name):
        dataset_size = self._db.get_dataset_size(name)
        offset = random.randint(0, max(0, dataset_size - QUICK_CLASSIFICATION_BATCH_SIZE))
        X = list()
        doc_ids = list()
        for doc in self._db.get_dataset_documents_without_labels(name, classifier_name, offset,
                                                                 QUICK_CLASSIFICATION_BATCH_SIZE):
            X.append(doc.text)
            doc_ids.append(doc.id)

        if len(X) == 0:
            cherrypy.response.status = 400
            return f'No unlabeled documents in dataset \'{name}\' for classifier \'{classifier_name}\''

        scores = self._db.score(classifier_name, X)
        positions_scores = list()
        for i, dict_ in enumerate(scores):
            probs = self._softmax(list(dict_.values()))
            probs.sort()
            diff = probs[-1] - probs[-2]
            positions_scores.append((i, diff))
        positions_scores.sort(key=lambda x: -x[1])
        return self._db.get_dataset_document_position_by_id(name, doc_ids[positions_scores[0][0]])

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def random_hidden_document_id(self, name, classifier_name):
        document_ids = [document.id for document in
                        self._db.get_dataset_documents_with_label(name, classifier_name, Label.HIDDEN_LABEL)]
        if document_ids:
            document_id = random.choice(document_ids)
            position = self._db.get_dataset_document_position_by_id(name, document_id)
            return position
        else:
            cherrypy.response.status = 400
            return f'No hidden documents in dataset \'{name}\' for classifier \'{classifier_name}\''

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def classify(self, **data):
        try:
            datasetname = data['name']
        except KeyError:
            cherrypy.response.status = 400
            return 'Must specify a dataset name'
        try:
            classifiers = data['classifiers']
        except KeyError:
            try:
                classifiers = data['classifiers[]']
            except KeyError:
                cherrypy.response.status = 400
                return 'Must specify a vector of names of classifiers'
        classifiers = np.atleast_1d(classifiers).tolist()

        last_update_time = self._db.get_most_recent_classifier_update_time(classifiers)
        dataset_update_time = self._db.get_dataset_last_update_time(datasetname)
        if last_update_time is None or last_update_time < dataset_update_time:
            last_update_time = dataset_update_time

        filename = 'dataset %s classified %s %s.csv' % (
            datasetname, "-".join(classifiers), str(last_update_time))
        filename = get_fully_portable_file_name(filename)
        fullpath = os.path.join(self._download_dir, filename)

        if self._db.classification_exists(fullpath):
            cherrypy.response.status = 409
            return 'An up-to-date classification is already available.'

        job_id = self._db.create_job(_classify,
                                     (self._db_connection_string, datasetname, classifiers, fullpath),
                                     description='classify dataset \'%s\' with %s' % (
                                         datasetname,
                                         ', '.join(['\'%s\'' % classifier for classifier in classifiers])))

        self._db.create_classification_job(datasetname, classifiers, job_id, fullpath)

        return [job_id]

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def classification_info(self, name, page=None, page_size=50):
        got_deleted = True
        result = None
        while got_deleted:
            got_deleted = False
            result = list()
            to_delete = list()
            if page is None:
                jobs = self._db.get_classification_jobs()
            else:
                jobs = self._db.get_classification_jobs(name)[int(page) * int(page_size):(int(page) + 1) * int(page_size)]
            for classification_job in jobs:
                classification_job_info = dict()
                classification_job_info['id'] = classification_job.id
                if (classification_job.filename is None or not os.path.exists(
                        classification_job.filename)) and classification_job.job is None:
                    to_delete.append(classification_job.id)
                    got_deleted = True
                    continue
                classification_job_info['dataset'] = name
                classification_job_info['classifiers'] = classification_job.classifiers
                classification_job_info['creation'] = str(classification_job.creation)
                if classification_job.job:
                    classification_job_info['status'] = classification_job.job.status
                    classification_job_info['completion'] = str(classification_job.job.completion)
                else:
                    classification_job_info['status'] = Job.status_done
                    classification_job_info['completion'] = str(os.path.getmtime(classification_job.filename))
                result.append(classification_job_info)

            for id in to_delete:
                self._db.delete_classification_job(id)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def classification_count(self, name):
        return str(len(list(self._db.get_classification_jobs(name))))

    @cherrypy.expose
    def classification_download(self, id):
        filename = self._db.get_classification_job_filename(int(id))
        if filename is None or not os.path.exists(filename):
            cherrypy.response.status = 404
            return "File not found"
        return serve_file(filename, "text/csv", "attachment")

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def classification_delete(self, id):
        try:
            filename = self._db.get_classification_job_filename(id)
            os.unlink(filename)
        except FileNotFoundError:
            pass
        self._db.delete_classification_job(id)
        return 'Ok'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def version(self):
        return "3.1.3 (db: %s)" % self._db.version()


def _classify(db_connection_string, datasetname, classifiers, fullpath):
    cherrypy.log('DatasetCollectionService._classify(datasetname="' + datasetname + '", classifiers="' + str(
        classifiers) + '", fullpath="' + fullpath + '")')
    with SQLAlchemyDB(db_connection_string) as db:
        tempfile = fullpath + '.tmp'
        try:
            with open(tempfile, 'w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file, lineterminator='\n')
                header = list()
                header.append('#id')
                header.append('text')
                for classifier in classifiers:
                    if db.classifier_exists(classifier):
                        header.append(
                            classifier + ' = (' + ', '.join(db.get_classifier_labels(classifier)) + ')')
                writer.writerow(header)

                batch_count = 0
                found = True
                while found:
                    found = False
                    X = list()
                    id = list()
                    for document in db.get_dataset_documents_by_name(datasetname, batch_count * MAX_BATCH_SIZE,
                                                                     MAX_BATCH_SIZE):
                        id.append(document.external_id)
                        X.append(document.text)
                    if len(X) > 0:
                        cols = list()
                        cols.append(id)
                        cols.append(X)
                        for classifier in classifiers:
                            cols.append(['%s:%s' % (classifier, y) for y in db.classify(classifier, X)])
                        for row in zip(*cols):
                            writer.writerow(row)
                        found = True
                    batch_count += 1
            try:
                os.unlink(fullpath)
            except FileNotFoundError:
                pass
            os.rename(tempfile, fullpath)
        except Exception as e:
            try:
                os.unlink(tempfile)
            except FileNotFoundError:
                pass
            try:
                os.unlink(fullpath)
            except FileNotFoundError:
                pass
            raise


def _create_dataset_documents(db_connection_string, dataset_name, filename):
    cherrypy.log(
        'DatasetCollectionService._create_dataset_documents(dataset_name="' + dataset_name + '", filename="' + filename + '")')
    with SQLAlchemyDB(db_connection_string) as db:
        if not db.dataset_exists(dataset_name):
            db.create_dataset(dataset_name)
        if csv.field_size_limit() < CSV_LARGE_FIELD:
            csv.field_size_limit(CSV_LARGE_FIELD)
        with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
            reader = csv.reader(file)
            external_ids_and_contents = list()
            for row in reader:
                if len(row) > 1:
                    document_name = row[0].strip()
                    if len(document_name) == 0 or document_name[0] == '#':
                        continue
                    content = row[1]
                    external_ids_and_contents.append((document_name, content))
                if len(external_ids_and_contents) >= MAX_BATCH_SIZE:
                    db.create_dataset_documents(dataset_name, external_ids_and_contents)
                    external_ids_and_contents = list()
            if len(external_ids_and_contents) > 0:
                db.create_dataset_documents(dataset_name, external_ids_and_contents)


def _delete_dataset(db_connection_string, name):
    cherrypy.log('DatasetCollectionService._delete_dataset(dname="' + name + '")')
    with SQLAlchemyDB(db_connection_string) as db:
        db.delete_dataset(name)
