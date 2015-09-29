from contextlib import contextmanager
import datetime
from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, PrimaryKeyConstraint, Text, \
    ForeignKeyConstraint, create_engine, PickleType
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, load_only, deferred
from sqlalchemy.orm.session import sessionmaker

__author__ = 'Andrea Esuli'

Base = declarative_base()

# CREATE TABLE IF NOT EXISTS classifier (
# name STRING PRIMARY KEY NOT NULL,
# model BLOB NOT NULL,
# creation TIMESTAMP DEFAULT (CURRENT_TIMESTAMP));
class Classifier(Base):
    __tablename__ = 'classifier'
    name = Column(String(50), primary_key=True)
    model = deferred(Column(PickleType(), nullable=False))
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True), onupdate=datetime.datetime.now)

    def __init__(self, name, model):
        self.name = name
        self.model = model


# CREATE TABLE IF NOT EXISTS class (
# name STRING NOT NULL,
# classifier STRING NOT NULL REFERENCES classifier (name) ON DELETE CASCADE ON UPDATE CASCADE,
# PRIMARY KEY (name, classifier));
class Label(Base):
    __tablename__ = 'label'
    name = Column(String(50), nullable=False)
    classifier = Column(String(50), ForeignKey('classifier.name', onupdate="CASCADE", ondelete="CASCADE"),
                        nullable=False)
    __table_args__ = (PrimaryKeyConstraint('classifier', 'name'), {})

    def __init__(self, name, classifier):
        self.name = name
        self.classifier = classifier


# CREATE TABLE IF NOT EXISTS document (
# id INTEGER PRIMARY KEY AUTOINCREMENT,
# text TEXT NOT NULL,
# creation TIMESTAMP DEFAULT (CURRENT_TIMESTAMP));
class Document(Base):
    __tablename__ = 'document'
    id = Column(Integer(), primary_key=True)
    text = Column(Text())
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)

    def __init__(self, text):
        self.text = text


# CREATE TABLE IF NOT EXISTS classification (
#  id INTEGER PRIMARY KEY AUTOINCREMENT,
#  document INTEGER NOT NULL,
#  class STRING NOT NULL,
#  classifier STRING NOT NULL,
#  creation TIMESTAMP DEFAULT (CURRENT_TIMESTAMP),
#  FOREIGN KEY (class, classifier) REFERENCES class (name, classifier) ON DELETE CASCADE ON UPDATE CASCADE);
class Classification(Base):
    __tablename__ = 'classification'
    id = Column(Integer(), primary_key=True)
    document = Column(Integer(), ForeignKey('document.id', onupdate='CASCADE', ondelete='CASCADE'),
                      nullable=False)
    classifier = Column(String(50))
    label = Column(String(50))
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    __table_args__ = (ForeignKeyConstraint([classifier, label],
                                           [Label.classifier, Label.name], ondelete='CASCADE', onupdate='CASCADE'),
                      {})

    def __init__(self, document, classifier, label):
        self.document = document
        self.classifier = classifier
        self.label = label


class SQLAlchemyDB(object):
    def __init__(self, name):
        engine = create_engine(name)
        Base.metadata.create_all(engine)
        self._sessionmaker = scoped_session(sessionmaker(bind=engine))

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
            return [classifier.name for classifier in session.query(Classifier).options(load_only('name')).all()]

    def classifier_exists(self, name):
        with self.session_scope() as session:
            return session.query(Classifier).filter(Classifier.name == name).count() > 0

    def create_classifier_model(self, name, classes, model):
        with self.session_scope() as session:
            classifier = Classifier(name, model)
            session = self._sessionmaker()
            session.add(classifier)
            session.flush()
            for class_ in classes:
                label = Label(class_, name)
                session.add(label)

    def get_classifier_model(self, name):
        with self.session_scope() as session:
            return session.query(Classifier).filter(Classifier.name == name)[0].model

    def get_classifier_classes(self, name):
        with self.session_scope() as session:
            return [label.name for label in session.query(Label).filter(Label.classifier == name).options(load_only('name'))]

    def get_classifier_creation_time(self, name):
        with self.session_scope() as session:
            return str(session.query(Classifier).filter(Classifier.name == name).options(load_only('creation'))[0].creation)

    def get_classifier_last_update_time(self, name):
        with self.session_scope() as session:
            return str(session.query(Classifier).filter(Classifier.name == name).options(load_only('creation'))[0].last_updated)

    def delete_classifier_model(self, name):
        with self.session_scope() as session:
            session.query(Classifier).filter(Classifier.name == name).delete()

    def update_classifier_model(self, name, model):
        with self.session_scope() as session:
            session.query(Classifier).filter(Classifier.name == name).options(load_only('name')).update({Classifier.model: model})

    def create_training_example(self, name, content, label):
        with self.session_scope() as session:
            document = Document(content)
            session.add(document)
            session.flush()
            classification = Classification(document.id, name, label)
            session.add(classification)

