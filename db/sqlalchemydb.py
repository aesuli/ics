from contextlib import contextmanager
import datetime
import random
import time
from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, PrimaryKeyConstraint, Text, \
    create_engine, PickleType
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, deferred, relationship
from sqlalchemy.orm.session import sessionmaker

__author__ = 'Andrea Esuli'

Base = declarative_base()


class Classifier(Base):
    __tablename__ = 'classifier'
    id = Column(Integer(), primary_key=True)
    name = Column(String(50), unique=True)
    model = deferred(Column(PickleType(), nullable=False))
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True), onupdate=datetime.datetime.now)

    def __init__(self, name, model):
        self.name = name
        self.model = model


class Label(Base):
    __tablename__ = 'label'
    id = Column(Integer(), unique=True)
    name = Column(String(50), nullable=False)
    classifier_id = Column(Integer(), ForeignKey('classifier.id', onupdate="CASCADE", ondelete="CASCADE"),
                           nullable=False)
    classifier = relationship('Classifier', backref='labels')
    __table_args__ = (PrimaryKeyConstraint('classifier_id', 'name'), {})

    def __init__(self, name, classifier_id):
        self.name = name
        self.classifier_id = classifier_id


class Dataset(Base):
    __tablename__ = 'dataset'
    id = Column(Integer(), primary_key=True)
    name = Column(String(50), unique=True)
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True), onupdate=datetime.datetime.now)

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
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)

    def __init__(self, text, collection_id, external_id=None):
        self.text = text
        self.collection_id = collection_id
        self.external_id = None


class Classification(Base):
    #TODO store source of classification (user/classifier -> which one?)
    __tablename__ = 'classification'
    id = Column(Integer(), primary_key=True)
    document_id = Column(Integer(), ForeignKey('document.id', onupdate='CASCADE', ondelete='CASCADE'),
                         nullable=False)
    document = relationship('Document', backref='classifications')
    label_id = Column(Integer(), ForeignKey('label.id', onupdate='CASCADE', ondelete='CASCADE'), nullable=False)
    label = relationship('Label', backref='labels')
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)

    def __init__(self, document_id, label_id):
        self.document_id = document_id
        self.label_id = label_id


class SQLAlchemyDB(object):
    MAXRETRY = 3
    TRAINING_DATASET = '_training_examples'

    def __init__(self, name):
        engine = create_engine(name)
        Base.metadata.create_all(engine)
        self._sessionmaker = scoped_session(sessionmaker(bind=engine))
        self._preload_data()

    def _preload_data(self):
        with self.session_scope() as session:
            if session.query(Dataset.name).filter(Dataset.name == SQLAlchemyDB.TRAINING_DATASET).count() < 1:
                session.add(Dataset(SQLAlchemyDB.TRAINING_DATASET))

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
            return session.query(Classifier.name).filter(Classifier.name == name).count() > 0

    def create_classifier_model(self, name, classes, model):
        with self.session_scope() as session:
            classifier = Classifier(name, model)
            session = self._sessionmaker()
            session.add(classifier)
            session.flush()
            for class_ in classes:
                label = Label(class_, classifier.id)
                session.add(label)

    def get_classifier_model(self, name):
        with self.session_scope() as session:
            return session.query(Classifier.model).filter(Classifier.name == name).first()[0]

    def get_classifier_classes(self, name):
        with self.session_scope() as session:
            return list(session.query(Label.name).join(Label.classifier).filter(Classifier.name == name))

    def get_classifier_creation_time(self, name):
        with self.session_scope() as session:
            return str(session.query(Classifier.creation).filter(Classifier.name == name).first())

    def get_classifier_last_update_time(self, name):
        with self.session_scope() as session:
            return str(session.query(Classifier.last_updated).filter(Classifier.name == name).first())

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

    def create_training_example(self, name, content, label):
        with self.session_scope() as session:
            document = session.query(Document).filter(Document.text == content).first()
            if document is None:
                try:
                    dataset_id = \
                    session.query(Dataset.id).filter(Dataset.name == SQLAlchemyDB.TRAINING_DATASET).one()[0]
                    document = Document(content, dataset_id)
                    session.add(document)
                    session.flush()
                except IntegrityError as ie:
                    session.rollback()
                    document = session.query(Document).filter(Document.text == content).first()
                    if document is None:
                        raise ie
            classifier_id = session.query(Classifier.id).filter(Classifier.name == name).first()
            label_id = session.query(Label.id).filter(Label.classifier_id == classifier_id).filter(
                Label.name == label).first()[0]
            classification = Classification(document.id, label_id)
            session.add(classification)

