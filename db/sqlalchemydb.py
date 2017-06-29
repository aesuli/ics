import datetime
import time
from contextlib import contextmanager
from uuid import uuid4

from passlib.hash import pbkdf2_sha256
from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Text, create_engine, PickleType, \
    UniqueConstraint, desc, exists
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, deferred, relationship, configure_mappers
from sqlalchemy.orm.session import sessionmaker

__author__ = 'Andrea Esuli'

Base = declarative_base()

classifier_name_length = 100
label_name_length = 50
dataset_name_length = 100
document_name_length = 100
salt_length = 20


class User(Base):
    __tablename__ = 'user'
    id = Column(Integer(), primary_key=True)
    name = Column(String(classifier_name_length), unique=True)
    salted_password = Column(String())
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)

    def __init__(self, name, password):
        self.name = name
        self.salted_password = pbkdf2_sha256.hash(password)

    def verify(self, password):
        return pbkdf2_sha256.verify(password, self.salted_password)

    def change_password(self, password):
        self.salted_password = pbkdf2_sha256.hash(password)


class Classifier(Base):
    __tablename__ = 'classifier'
    id = Column(Integer(), primary_key=True)
    name = Column(String(classifier_name_length), unique=True)
    model = deferred(Column(PickleType()))
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True), default=datetime.datetime.now, onupdate=datetime.datetime.now)

    def __init__(self, name, model):
        self.name = name
        self.model = model


class Label(Base):
    __tablename__ = 'label'
    id = Column(Integer(), primary_key=True)
    name = Column(String(label_name_length), nullable=False)
    classifier_id = Column(Integer(), ForeignKey('classifier.id', onupdate="CASCADE", ondelete="CASCADE"),
                           nullable=False)
    classifier = relationship('Classifier', backref='labels')
    __table_args__ = (UniqueConstraint('classifier_id', 'name'), {})

    def __init__(self, name, classifier_id):
        self.name = name
        self.classifier_id = classifier_id


class Dataset(Base):
    __tablename__ = 'dataset'
    id = Column(Integer(), primary_key=True)
    name = Column(String(dataset_name_length), unique=True)
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True), default=datetime.datetime.now, onupdate=datetime.datetime.now)

    def __init__(self, name):
        self.name = name


class Document(Base):
    __tablename__ = 'document'
    id = Column(Integer(), primary_key=True)
    external_id = Column(String(document_name_length))
    dataset_id = Column(Integer(), ForeignKey('dataset.id', onupdate='CASCADE', ondelete='CASCADE'),
                        nullable=False)
    dataset = relationship('Dataset', backref='documents')
    text = Column(Text())
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now, onupdate=datetime.datetime.now)

    def __init__(self, text, dataset_id, external_id=None):
        self.text = text
        self.dataset_id = dataset_id
        self.external_id = external_id


class Classification(Base):
    __tablename__ = 'classification'
    id = Column(Integer(), primary_key=True)
    document_id = Column(Integer(), ForeignKey('document.id', onupdate='CASCADE', ondelete='CASCADE'),
                         nullable=False)
    document = relationship('Document', backref='classifications')
    label_id = Column(Integer(), ForeignKey('label.id', onupdate='CASCADE', ondelete='CASCADE'), nullable=False)
    label = relationship('Label', backref='classifications')
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)

    def __init__(self, document_id, label_id):
        self.document_id = document_id
        self.label_id = label_id


class Job(Base):
    __tablename__ = 'job'
    id = Column(Integer(), primary_key=True)
    description = Column(Text())
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    start = Column(DateTime(timezone=True))
    completion = Column(DateTime(timezone=True))
    status_pending = 'pending'
    status_done = 'done'
    status_error = 'error'
    status_running = 'running'
    status = Column(String(10), default=status_pending)

    def __init__(self, description):
        self.description = description


class ClassificationJob(Base):
    __tablename__ = 'classificationjob'
    id = Column(Integer(), primary_key=True)
    dataset_id = Column(Integer(), ForeignKey('dataset.id', onupdate='CASCADE', ondelete='CASCADE'),
                        nullable=False)
    dataset = relationship('Dataset', backref='classifications')
    job_id = Column(Integer(), ForeignKey('job.id', onupdate='CASCADE', ondelete='CASCADE'),
                    nullable=False)
    job = relationship('Job', backref='classification_job')
    classifiers = Column(Text())
    filename = Column(Text())

    def __init__(self, dataset_id, classifiers, job_id, filename):
        self.dataset_id = dataset_id
        self.classifiers = classifiers
        self.job_id = job_id
        self.filename = filename


class Lock(Base):
    __tablename__ = 'lock'
    name = Column(String(classifier_name_length + 20), primary_key=True)
    locker = Column(String(40))

    def __init__(self, name, locker):
        self.name = name
        self.locker = locker


class SQLAlchemyDB(object):
    _INTERNAL_TRAINING_DATASET = '_internal_training_dataset'
    _ADMIN_NAME = 'admin'
    _ADMIN_PASSWORD = 'adminadmin'

    def __init__(self, name):
        self._engine = create_engine(name)
        Base.metadata.create_all(self._engine)
        self._sessionmaker = scoped_session(sessionmaker(bind=self._engine))
        configure_mappers()
        self._preload_data()

    def _preload_data(self):
        with self.session_scope() as session:
            if not session.query(exists().where(Dataset.name == SQLAlchemyDB._INTERNAL_TRAINING_DATASET)).scalar():
                session.add(Dataset(SQLAlchemyDB._INTERNAL_TRAINING_DATASET))
            if not session.query(exists().where(User.name == SQLAlchemyDB._ADMIN_NAME)).scalar():
                session.add(User(SQLAlchemyDB._ADMIN_NAME, SQLAlchemyDB._ADMIN_PASSWORD))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._sessionmaker.session_factory.close_all()
        self._engine.dispose()

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self._sessionmaker()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def admin_name():
        return SQLAlchemyDB._ADMIN_NAME

    def create_user(self, name, password):
        with self.session_scope() as session:
            user = User(name, password)
            session.add(user)
            session.commit()

    def user_exists(self, name):
        with self.session_scope() as session:
            return session.query(exists().where(User.name == name)).scalar()

    def verify_user(self, name, password):
        with self.session_scope() as session:
            user = session.query(User).filter(User.name == name).scalar()
            return user.verify(password)

    def delete_user(self, name):
        with self.session_scope() as session:
            session.query(User).filter(User.name == name).delete()

    def change_password(self, name, password):
        with self.session_scope() as session:
            user = session.query(User).filter(User.name == name).scalar()
            user.change_password(password)

    def user_names(self):
        with self.session_scope() as session:
            return list(session.query(User.name).order_by(User.name).all())

    def get_user_creation_time(self, name):
        with self.session_scope() as session:
            return session.query(User.creation).filter(User.name == name).scalar()

    def classifier_names(self):
        with self.session_scope() as session:
            return list(session.query(Classifier.name).order_by(Classifier.name).all())

    def classifier_exists(self, name):
        with self.session_scope() as session:
            return session.query(exists().where(Classifier.name == name)).scalar()

    def create_classifier(self, name, labels, model=None):
        with self.session_scope() as session:
            classifier = Classifier(name, model)
            session.add(classifier)
            session.flush()
            for label in labels:
                label_obj = Label(label, classifier.id)
                session.add(label_obj)

    def get_classifier_model(self, name):
        with self.session_scope() as session:
            return session.query(Classifier.model).filter(Classifier.name == name).scalar()

    def get_classifier_labels(self, name):
        with self.session_scope() as session:
            return list(x for (x,) in session.query(Label.name).order_by(Label.name).join(Label.classifier).filter(
                Classifier.name == name))

    def get_classifier_creation_time(self, name):
        with self.session_scope() as session:
            return session.query(Classifier.creation).filter(Classifier.name == name).scalar()

    def get_classifier_last_update_time(self, name):
        with self.session_scope() as session:
            return session.query(Classifier.last_updated).filter(Classifier.name == name).scalar()

    def rename_classifier(self, name, newname):
        with self.session_scope() as session:
            classifier = session.query(Classifier).filter(Classifier.name == name).scalar()
            if classifier is None:
                return
            classifier.name = newname

    def delete_classifier(self, name):
        with self.session_scope() as session:
            session.query(Classifier).filter(Classifier.name == name).delete()

    def update_classifier_model(self, name, model):
        with self.session_scope() as session:
            try:
                session.query(Classifier).filter(Classifier.name == name).update({Classifier.model: model})
            except (OperationalError, MemoryError) as e:
                session.rollback()
                raise e

    def create_training_example(self, classifier_name, content, label):
        with self.session_scope() as session:
            training_dataset = session.query(Dataset).filter(
                Dataset.name == SQLAlchemyDB._INTERNAL_TRAINING_DATASET).one()
            document = session.query(Document).filter(Document.text == content).filter(
                training_dataset.id == Document.dataset_id).scalar()
            if document is None:
                document = Document(content, training_dataset.id)
                session.add(document)
            training_dataset.last_updated = document.creation
            session.commit()

        with self.session_scope() as session:
            training_dataset = session.query(Dataset).filter(
                Dataset.name == SQLAlchemyDB._INTERNAL_TRAINING_DATASET).one()
            document = session.query(Document).filter(Document.text == content).filter(
                training_dataset.id == Document.dataset_id).scalar()
            label_id = session.query(Label.id).filter(Classifier.name == classifier_name).filter(
                Label.classifier_id == Classifier.id).filter(
                Label.name == label).scalar()

            classification = session.query(Classification).filter(Classification.document_id == document.id).join(
                Classification.label).filter(Classifier.name == classifier_name).filter(
                Label.classifier_id == Classifier.id).scalar()

            if classification is None:
                classification = Classification(document.id, label_id)
                session.add(classification)
            else:
                classification.label_id = label_id

    def classify(self, classifier_name, X):
        clf = self.get_classifier_model(classifier_name)
        y = list()
        if clf is None:
            default_label = self.get_classifier_labels(classifier_name)[0]
        for x in X:
            label = self.get_label(classifier_name, x)
            if label is not None:
                y.append(label)
            elif clf is not None:
                y.append(clf.predict([x])[0])
            else:
                y.append(default_label)
        return y

    def get_label(self, classifier_name, content):
        with self.session_scope() as session:
            return session.query(Label.name).filter(Document.text == content).filter(
                Classification.document_id == Document.id).filter(Classifier.name == classifier_name).filter(
                Label.classifier_id == Classifier.id).filter(Label.id == Classification.label_id).order_by(
                desc(Classification.creation)).scalar()

    def rename_classifier_label(self, classifier_name, label_name, newname):
        with self.session_scope() as session:
            label = session.query(Label).filter(Label.name == label_name).join(Label.classifier).filter(
                Classifier.name == classifier_name).scalar()
            if label is None:
                return
            label.name = newname

    def dataset_names(self):
        with self.session_scope() as session:
            return list(
                session.query(Dataset.name).order_by(Dataset.name).filter(
                    Dataset.name != SQLAlchemyDB._INTERNAL_TRAINING_DATASET))

    def dataset_exists(self, name):
        with self.session_scope() as session:
            return session.query(exists().where(Dataset.name == name)).scalar()

    def create_dataset(self, name):
        with self.session_scope() as session:
            dataset = Dataset(name)
            session.add(dataset)

    def rename_dataset(self, name, newname):
        if name == SQLAlchemyDB._INTERNAL_TRAINING_DATASET:
            return
        with self.session_scope() as session:
            dataset = session.query(Dataset).filter(Dataset.name == name).scalar()
            if dataset is None:
                return
            dataset.name = newname

    def delete_dataset(self, name):
        if name == SQLAlchemyDB._INTERNAL_TRAINING_DATASET:
            return
        with self.session_scope() as session:
            session.query(Dataset).filter(Dataset.name == name).delete()

    def get_dataset_creation_time(self, name):
        with self.session_scope() as session:
            return session.query(Dataset.creation).filter(Dataset.name == name).scalar()

    def get_dataset_last_update_time(self, name):
        with self.session_scope() as session:
            return session.query(Dataset.last_updated).filter(Dataset.name == name).scalar()

    def get_dataset_size(self, name):
        with self.session_scope() as session:
            return session.query(Dataset.documents).filter(Dataset.name == name).filter(
                Document.dataset_id == Dataset.id).count()

    def create_document(self, dataset_name, external_id, content):
        with self.session_scope() as session:
            dataset = session.query(Dataset).filter(Dataset.name == dataset_name).one()
            document = session.query(Document).filter(Document.dataset_id == dataset.id).filter(
                Document.external_id == external_id).scalar()
            if document is None:
                document = Document(content, dataset.id, external_id)
                session.add(document)
            else:
                document.text = content
            dataset.last_updated = datetime.datetime.now()

    def get_classifier_examples_count(self, name):
        with self.session_scope() as session:
            return session.query(Classification.id).filter(Classifier.name == name).filter(
                Label.classifier_id == Classifier.id).filter(Classification.label_id == Label.id).count()

    def get_classifier_examples_with_label_count(self, name, label):
        with self.session_scope() as session:
            return session.query(Classification.id).filter(Classifier.name == name).filter(
                Label.classifier_id == Classifier.id).filter(Classification.label_id == Label.id).filter(
                Label.name == label).count()

    def get_classifier_examples(self, name, offset=0, limit=None):
        with self.session_scope() as session:
            return session.query(Classification).order_by(Classification.creation).filter(
                Classifier.name == name).filter(Label.classifier_id == Classifier.id).filter(
                Classification.label_id == Label.id).offset(offset).limit(limit)

    def get_classifier_examples_with_label(self, name, label, offset=0, limit=None):
        with self.session_scope() as session:
            return session.query(Classification).order_by(Classification.creation).filter(
                Classifier.name == name).filter(Label.classifier_id == Classifier.id).filter(
                Classification.label_id == Label.id).filter(Label.name == label).offset(offset).limit(limit)

    def get_dataset_documents(self, name):
        with self.session_scope() as session:
            return session.query(Document).order_by(Document.external_id).filter(Dataset.name == name).filter(
                Document.dataset_id == Dataset.id)

    def get_dataset_document(self, name, position):
        with self.session_scope() as session:
            document = session.query(Document).filter(Dataset.name == name).filter(
                Document.dataset_id == Dataset.id).offset(position).limit(1).scalar()
            if document is not None:
                session.expunge(document)
            return document

    def get_jobs(self, starttime=None):
        if starttime is None:
            starttime = datetime.datetime.now() - datetime.timedelta(days=1)
        with self.session_scope() as session:
            return session.query(Job).order_by(Job.creation.desc()).filter(Job.creation > starttime)

    def create_job(self, description):
        with self.session_scope() as session:
            job = Job(description)
            session.add(job)
            session.commit()
            return job.id

    def set_job_completion_time(self, job_id, completion=datetime.datetime.now()):
        with self.session_scope() as session:
            job = session.query(Job).filter(Job.id == job_id).one()
            job.completion = completion

    def set_job_start_time(self, job_id, start=datetime.datetime.now()):
        with self.session_scope() as session:
            job = session.query(Job).filter(Job.id == job_id).one()
            job.start = start

    def set_job_status(self, job_id, status):
        with self.session_scope() as session:
            job = session.query(Job).filter(Job.id == job_id).one()
            job.status = status

    def get_most_recent_classifier_update_time(self, classifiers):
        with self.session_scope() as session:
            return session.query(Classifier.last_updated).filter(Classifier.name.in_(classifiers)).order_by(
                Classifier.last_updated.desc()).first()[0]

    def create_classification_job(self, datasetname, classifiers, job_id, fullpath):
        with self.session_scope() as session:
            dataset_id = session.query(Dataset.id).filter(Dataset.name == datasetname).scalar()
            classification_job = ClassificationJob(dataset_id, ', '.join(classifiers), job_id, fullpath)
            session.add(classification_job)

    def get_classification_jobs(self, name):
        with self.session_scope() as session:
            return session.query(ClassificationJob).filter(ClassificationJob.dataset_id == Dataset.id).filter(
                Dataset.name == name).join(Job).order_by(Job.creation.desc())

    def get_classification_job_filename(self, id):
        with self.session_scope() as session:
            return session.query(ClassificationJob.filename).filter(ClassificationJob.id == id).scalar()

    def delete_classification_job(self, id):
        with self.session_scope() as session:
            classification_job = session.query(ClassificationJob).filter(
                ClassificationJob.id == id).scalar()
            session.delete(classification_job)

    def delete_job(self, id):
        with self.session_scope() as session:
            session.query(Job).filter(Job.id == id).delete()

    def classification_exists(self, filename):
        with self.session_scope() as session:
            return session.query(exists().where(ClassificationJob.filename == filename)).scalar()

    def job_exists(self, id):
        with self.session_scope() as session:
            return session.query(exists().where(Job.id == id)).scalar()

    def acquire_lock(self, name, locker, poll_interval=1):
        with self.session_scope() as session:
            locked = False
            while not locked:
                try:
                    lock = Lock(name, locker)
                    session.add(lock)
                    session.commit()
                    locked = True
                except:
                    session.rollback()
                    time.sleep(poll_interval)

    def release_lock(self, name, locker):
        with self.session_scope() as session:
            lock = session.query(Lock).filter(Lock.name == name).filter(Lock.locker == locker).first()
            if lock is not None:
                session.delete(lock)

    @staticmethod
    def version():
        return "0.5.1"


class DBLock(object):
    def __init__(self, db, name):
        self._db = db
        self._name = name
        self._uuid = uuid4()

    def acquire(self):
        self._db.acquire_lock(self._name, str(self._uuid))

    def release(self):
        self._db.release_lock(self._name, str(self._uuid))

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        self.release()
        return None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return None

    def __del__(self):
        self.release()
        return None
