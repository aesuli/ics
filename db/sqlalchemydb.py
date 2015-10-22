from contextlib import contextmanager
import datetime
import random
import time
import traceback
from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Text, create_engine, PickleType, \
    UniqueConstraint, desc, exists
from sqlalchemy.exc import IntegrityError, OperationalError
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


class SQLAlchemyDB(object):
    MAXRETRY = 3
    INTERNAL_TRAINING_DATASET = '_internal_training_dataset'

    def __init__(self, name):
        engine = create_engine(name)
        Base.metadata.create_all(engine)
        self._sessionmaker = scoped_session(sessionmaker(bind=engine))
        configure_mappers()
        self._preload_data()

    def _preload_data(self):
        with self.session_scope() as session:
            if not session.query(exists().where(Dataset.name == SQLAlchemyDB.INTERNAL_TRAINING_DATASET)).scalar():
                session.add(Dataset(SQLAlchemyDB.INTERNAL_TRAINING_DATASET))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._sessionmaker.close_all()

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
            return list(session.query(Classifier.name).all())

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
            return list(x for (x,) in session.query(Label.name).join(Label.classifier).filter(Classifier.name == name))

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
                    if retries == SQLAlchemyDB.MAXRETRY:
                        raise e

    def create_training_example(self, classifier_name, content, label):
        with self.session_scope() as session:
            training_dataset = session.query(Dataset).filter(
                Dataset.name == SQLAlchemyDB.INTERNAL_TRAINING_DATASET).one()
            document = session.query(Document).filter(Document.text == content).filter(
                training_dataset.id == Document.dataset_id).first()  # TODO one_or_none()
            if document is None:
                try:
                    document = Document(content, training_dataset.id)
                    session.add(document)
                    session.flush()
                except IntegrityError as ie:
                    session.rollback()
                    document = session.query(Document).filter(Document.text == content).filter(
                        training_dataset.id == Document.dataset_id).one()
                    if document is None:
                        raise ie
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

            training_dataset.last_updated = datetime.datetime.now()

    def get_label(self, classifier_name, content):
        with self.session_scope() as session:
            return session.query(Label.name).filter(Document.text == content).filter(
                Classification.document_id == Document.id).filter(Classifier.name == classifier_name) \
                .filter(Label.classifier_id == Classifier.id).filter(Label.id == Classification.label_id).order_by(
                desc(Classification.creation)).scalar()

    def dataset_names(self):
        with self.session_scope() as session:
            return list(
                session.query(Dataset.name).filter(Dataset.name != SQLAlchemyDB.INTERNAL_TRAINING_DATASET))

    def dataset_exists(self, name):
        with self.session_scope() as session:
            return session.query(exists().where(Dataset.name == name)).scalar()

    def create_dataset(self, name):
        with self.session_scope() as session:
            dataset = Dataset(name)
            session.add(dataset)

    def delete_dataset(self, name):
        if name == SQLAlchemyDB.INTERNAL_TRAINING_DATASET:
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
            return session.query(Classification).filter(Classifier.name == name).filter(
                Label.classifier_id == Classifier.id).filter(Classification.label_id == Label.id)

    def get_dataset_documents(self, name):
        with self.session_scope() as session:
            return session.query(Document).filter(Dataset.name == name).filter(
                Document.dataset_id == Dataset.id)

    def version(self):
        return "0.1.0"


