from contextlib import contextmanager
import datetime
import random
import time
import threading

from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Text, create_engine, PickleType, \
    UniqueConstraint, desc, exists
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, deferred, relationship, configure_mappers
from sqlalchemy.orm.session import sessionmaker

__author__ = 'Andrea Esuli'

Base = declarative_base()


class Classifier(Base):
    __tablename__ = 'classifier'
    id = Column(Integer(), primary_key=True)
    name = Column(String(50), unique=True)
    model = deferred(Column(PickleType(), nullable=False))
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True), default=datetime.datetime.now, onupdate=datetime.datetime.now)

    def __init__(self, name, model):
        self.name = name
        self.model = model


class Label(Base):
    __tablename__ = 'label'
    id = Column(Integer(), primary_key=True)
    name = Column(String(50), nullable=False)
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
    name = Column(String(50), unique=True)
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True), default=datetime.datetime.now, onupdate=datetime.datetime.now)

    def __init__(self, name):
        self.name = name


class Document(Base):
    __tablename__ = 'document'
    id = Column(Integer(), primary_key=True)
    external_id = Column(String(50))
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
    description = Column(String(250))
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    start = Column(DateTime(timezone=True))
    completion = Column(DateTime(timezone=True))
    status = Column(String(10), default='pending')

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
    job = relationship('Job')
    classifiers = Column(Text())
    filename = Column(Text())

    def __init__(self, dataset_id, classifiers, job_id, filename):
        self.dataset_id = dataset_id
        self.classifiers = classifiers
        self.job_id = job_id
        self.filename = filename


class SQLAlchemyDB(object):
    _MAXRETRY = 3
    _INTERNAL_TRAINING_DATASET = '_internal_training_dataset'
    _TRAINING_EXAMPLE_CREATION_LOCK = threading.Lock()

    def __init__(self, name):
        engine = create_engine(name)
        Base.metadata.create_all(engine)
        self._sessionmaker = scoped_session(sessionmaker(bind=engine))
        configure_mappers()
        self._preload_data()

    def _preload_data(self):
        with self.session_scope() as session:
            if not session.query(exists().where(Dataset.name == SQLAlchemyDB._INTERNAL_TRAINING_DATASET)).scalar():
                session.add(Dataset(SQLAlchemyDB._INTERNAL_TRAINING_DATASET))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._sessionmaker.session_factory.close_all()

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

    def classifier_names(self):
        with self.session_scope() as session:
            return list(session.query(Classifier.name).order_by(Classifier.name).all())

    def classifier_exists(self, name):
        with self.session_scope() as session:
            return session.query(exists().where(Classifier.name == name)).scalar()

    def create_classifier(self, name, classes, model):
        with self.session_scope() as session:
            classifier = Classifier(name, model)
            session.add(classifier)
            session.flush()
            for class_ in classes:
                label = Label(class_, classifier.id)
                session.add(label)

    def get_classifier_model(self, name):
        with self.session_scope() as session:
            return session.query(Classifier.model).filter(Classifier.name == name).scalar()

    def get_classifier_classes(self, name):
        with self.session_scope() as session:
            return list(x for (x,) in session.query(Label.name).order_by(Label.name).join(Label.classifier).filter(
                Classifier.name == name))

    def get_classifier_creation_time(self, name):
        with self.session_scope() as session:
            return str(session.query(Classifier.creation).filter(Classifier.name == name).scalar())

    def get_classifier_last_update_time(self, name):
        with self.session_scope() as session:
            return str(session.query(Classifier.last_updated).filter(Classifier.name == name).scalar())

    def delete_classifier_model(self, name):
        with self.session_scope() as session:
            session.query(Classifier).filter(Classifier.name == name).delete()

    def update_classifier_model(self, name, model):
        with self.session_scope() as session:
            # TODO fix memory errors handling using a queue or 64 bit or...
            retries = 0
            while True:
                try:
                    session.query(Classifier).filter(Classifier.name == name).update({Classifier.model: model})
                    break
                except (OperationalError, MemoryError) as e:
                    session.rollback()
                    time.sleep(random.uniform(0.1, 1))
                    ++retries
                    if retries == SQLAlchemyDB._MAXRETRY:
                        raise e

    def create_training_example(self, classifier_name, content, label):
        with self.session_scope() as session:
            training_dataset = session.query(Dataset).filter(
                Dataset.name == SQLAlchemyDB._INTERNAL_TRAINING_DATASET).one()
            with SQLAlchemyDB._TRAINING_EXAMPLE_CREATION_LOCK:
                document = session.query(Document).filter(Document.text == content).filter(
                    training_dataset.id == Document.dataset_id).first()  # TODO one_or_none()
                if document is None:
                    document = Document(content, training_dataset.id)
                    session.add(document)
                training_dataset.last_updated = document.creation
                session.commit()

        with self.session_scope() as session:
            training_dataset = session.query(Dataset).filter(
                Dataset.name == SQLAlchemyDB._INTERNAL_TRAINING_DATASET).one()
            document = session.query(Document).filter(Document.text == content).filter(
                training_dataset.id == Document.dataset_id).first()  # TODO one_or_none()
            label_id = session.query(Label.id).filter(Classifier.name == classifier_name).filter(
                Label.classifier_id == Classifier.id).filter(
                Label.name == label).scalar()

            classification = session.query(Classification).filter(Classification.document_id == document.id).join(
                Classification.label).filter(Classifier.name == classifier_name).filter(
                Label.classifier_id == Classifier.id)

            classification = classification.first()  # TODO one_or_none()

            if classification is None:
                classification = Classification(document.id, label_id)
                session.add(classification)
            else:
                classification.label_id = label_id

    def classify(self, classifier_name, X):
        clf = self.get_classifier_model(classifier_name)
        y = list()
        for x in X:
            label = self.get_label(classifier_name, x)
            if label is not None:
                y.append(label)
            else:
                y.append(clf.predict([x])[0])
        return y

    def get_label(self, classifier_name, content):
        with self.session_scope() as session:
            return session.query(Label.name).filter(Document.text == content).filter(
                Classification.document_id == Document.id).filter(Classifier.name == classifier_name) \
                .filter(Label.classifier_id == Classifier.id).filter(Label.id == Classification.label_id).order_by(
                desc(Classification.creation)).scalar()

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

    def delete_dataset(self, name):
        if name == SQLAlchemyDB._INTERNAL_TRAINING_DATASET:
            return
        with self.session_scope() as session:
            session.query(Dataset).filter(Dataset.name == name).delete()

    def get_dataset_creation_time(self, name):
        with self.session_scope() as session:
            return str(session.query(Dataset.creation).filter(Dataset.name == name).scalar())

    def get_dataset_last_update_time(self, name):
        with self.session_scope() as session:
            return str(session.query(Dataset.last_updated).filter(Dataset.name == name).scalar())

    def get_dataset_size(self, name):
        with self.session_scope() as session:
            return session.query(Dataset.documents).filter(Dataset.name == name).filter(
                Document.dataset_id == Dataset.id).count()

    def create_document(self, dataset_name, external_id, content):
        with self.session_scope() as session:
            dataset = session.query(Dataset).filter(Dataset.name == dataset_name).one()
            document = session.query(Document).filter(Document.dataset_id == dataset.id).filter(
                Document.external_id == external_id).first()  # TODO one_or_none()
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

    def get_classifier_examples(self, name):
        with self.session_scope() as session:
            return session.query(Classification).order_by(Classification.creation).filter(
                Classifier.name == name).filter(Label.classifier_id == Classifier.id).filter(
                Classification.label_id == Label.id)

    def get_dataset_documents(self, name):
        with self.session_scope() as session:
            return session.query(Document).order_by(Document.external_id).filter(Dataset.name == name).filter(
                Document.dataset_id == Dataset.id)

    def get_dataset_document(self, name, position):
        with self.session_scope() as session:
            document = session.query(Document).filter(Dataset.name == name).filter(
                Document.dataset_id == Dataset.id).offset(int(position)).first()
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
            return session.query(Classifier.last_updated).filter(Classifier.name in classifiers).order_by(
                Classifier.last_updated.desc()).first()

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
                ClassificationJob.id == id).first()  # TODO one_or_none
            session.delete(classification_job.job)

    def delete_job(self, id):
        with self.session_scope() as session:
            session.query(Job).filter(Job.id == id).delete()

    def classification_exists(self, filename):
        with self.session_scope() as session:
            return session.query(exists().where(ClassificationJob.filename == filename)).scalar()

    def job_exists(self, id):
        with self.session_scope() as session:
            return session.query(exists().where(Job.id == id)).scalar()

    @staticmethod
    def version():
        return "0.2.3"

