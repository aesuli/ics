import datetime
import os
import re
import secrets
import time
from contextlib import contextmanager
from enum import Enum
from hashlib import md5
from uuid import uuid4

from passlib.hash import pbkdf2_sha256
from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Text, \
    create_engine, PickleType, UniqueConstraint, exists, not_, and_, func, \
    Enum as SQLEnum, Boolean
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import OperationalError, ResourceClosedError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, deferred, relationship, \
    configure_mappers, backref
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.sql.functions import count

from ics.classifier.classifier import create_classifier_model, YES_LABEL

__author__ = 'Andrea Esuli'

Base = declarative_base()

_ADMIN_NAME = 'admin'
_ADMIN_PASSWORD = 'adminadmin'
_NO_HOURLY_LIMIT = -1
_NO_REQUEST_LIMIT = -1

classifier_name_length = 80
classifier_description_length = 300
user_name_length = 50
key_name_length = 50
ipaddress_length = 45
key_length = 30
label_name_length = 50
dataset_name_length = 100
dataset_description_length = 300
document_name_length = 100


class LabelSource(Enum):
    HUMAN_LABEL = '\\H'
    MACHINE_LABEL = '\\M'


class ClassificationMode(Enum):
    MULTI_LABEL = "Multi-label"
    SINGLE_LABEL = "Single-label"


class Tracker(Base):
    __tablename__ = 'tracker'
    id = Column(Integer(), primary_key=True)
    hourly_limit = Column(Integer())
    current_request_counter = Column(Integer(), default=0)
    counter_time_span = Column(Integer(), default=int(time.time() / 3600))
    total_request_counter = Column(Integer(), default=0)
    request_limit = Column(Integer(), default=0)
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True),
                          default=datetime.datetime.now)
    type = Column(String(20))

    __mapper_args__ = {
        'polymorphic_identity': 'tracker',
        'polymorphic_on': type
    }

    def __init__(self, hourly_limit: int, request_limit: int):
        self.hourly_limit = int(hourly_limit)
        self.request_limit = int(request_limit)

    def check_and_count_request(self, cost=1):
        if 0 <= self.request_limit <= self.total_request_counter:
            return False
        if self.hourly_limit < 0:
            self.current_request_counter += cost
            self.total_request_counter += cost
            self.last_updated = datetime.datetime.now()
            return True
        current_time_span = int(time.time() / 3600)
        if self.counter_time_span < current_time_span:
            self.current_request_counter = 0
        self.counter_time_span = current_time_span
        if self.current_request_counter < self.hourly_limit:
            self.current_request_counter += cost
            self.total_request_counter += cost
            self.last_updated = datetime.datetime.now()
            return True
        else:
            return False

    def check_current_request_counter(self):
        current_time_span = int(time.time() / 3600)
        if self.counter_time_span < current_time_span:
            self.current_request_counter = 0
        self.counter_time_span = current_time_span
        return self.current_request_counter


class KeyTracker(Tracker):
    __tablename__ = 'key'
    id = Column(Integer, ForeignKey('tracker.id', onupdate='CASCADE',
                                    ondelete='CASCADE'), primary_key=True)
    key = Column(String(key_length * 2), unique=True)
    name = Column(String(key_name_length), unique=True)
    __mapper_args__ = {
        'polymorphic_identity': 'keytracker',
    }

    def __init__(self, name: str, hourly_limit, request_limit):
        super().__init__(hourly_limit, request_limit)
        self.key = secrets.token_hex(key_length)
        self.name = name


class IP(Tracker):
    __tablename__ = 'ip'
    id = Column(Integer, ForeignKey('tracker.id', onupdate='CASCADE',
                                    ondelete='CASCADE'), primary_key=True)
    ip = Column(String(ipaddress_length), unique=True)
    __mapper_args__ = {
        'polymorphic_identity': 'iptracker',
    }

    def __init__(self, ip: str, hourly_limit, request_limit):
        super().__init__(hourly_limit, request_limit)
        self.ip = ip.strip()


class User(Tracker):
    __tablename__ = 'user'
    id = Column(Integer, ForeignKey('tracker.id', onupdate='CASCADE',
                                    ondelete='CASCADE'), primary_key=True)
    name = Column(String(user_name_length), unique=True)
    salted_password = Column(String())
    __mapper_args__ = {
        'polymorphic_identity': 'user',
    }

    def __init__(self, name: str, password: str, hourly_limit=_NO_HOURLY_LIMIT,
                 request_limit=_NO_REQUEST_LIMIT):
        super().__init__(hourly_limit, request_limit)
        if re.match(r'^\w+$', name) is None:
            raise ValueError('Invalid username')
        self.name = name
        self.salted_password = pbkdf2_sha256.hash(password)

    def verify(self, password: str):
        return pbkdf2_sha256.verify(password, self.salted_password)

    def change_password(self, password: str):
        self.salted_password = pbkdf2_sha256.hash(password)


class Dataset(Base):
    __tablename__ = 'dataset'
    id = Column(Integer(), primary_key=True)
    name = Column(String(dataset_name_length), unique=True)
    description = Column(String(dataset_description_length),
                         default="No description")
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True),
                          default=datetime.datetime.now,
                          onupdate=datetime.datetime.now)

    def __init__(self, name: str):
        self.name = name


class DatasetDocument(Base):
    __tablename__ = 'dataset_document'
    id = Column(Integer(), primary_key=True, index=True)
    external_id = Column(String(document_name_length))
    dataset_id = Column(Integer(), ForeignKey('dataset.id', onupdate='CASCADE',
                                              ondelete='CASCADE'))
    dataset = relationship('Dataset',
                           backref=backref('documents',
                                           cascade="all, delete-orphan",
                                           passive_deletes=True))
    text = Column(Text())
    md5 = Column(Text(), index=True)
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now,
                      onupdate=datetime.datetime.now)
    __table_args__ = (UniqueConstraint('dataset_id', 'external_id'),)

    def __init__(self, text: str, dataset_id: int, external_id: int = None):
        self.text = text
        self.md5 = md5(text.encode('utf-8')).hexdigest()
        self.dataset_id = dataset_id
        self.external_id = external_id


class Classifier(Base):
    __tablename__ = 'classifier'
    id = Column(Integer(), primary_key=True)
    name = Column(String(classifier_name_length), unique=True)
    description = Column(String(classifier_description_length),
                         default="No description")
    public = Column(Boolean(), default=False)
    preferred_classification_mode = Column(SQLEnum(ClassificationMode))
    model = deferred(Column(PickleType()))
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True),
                          default=datetime.datetime.now,
                          onupdate=datetime.datetime.now)

    def __init__(self, name: str, classification_mode: ClassificationMode):
        self.name = name
        self.model = None
        self.preferred_classification_mode = classification_mode


class Label(Base):
    __tablename__ = 'label'
    id = Column(Integer(), primary_key=True)
    name = Column(String(label_name_length), nullable=False)
    classifier_id = Column(Integer(),
                           ForeignKey('classifier.id', onupdate='CASCADE',
                                      ondelete='CASCADE'))
    classifier = relationship('Classifier', backref=backref('labels',
                                                            cascade="all, delete-orphan",
                                                            passive_deletes=True))
    __table_args__ = (UniqueConstraint('classifier_id', 'name'),)

    def __init__(self, name: str, classifier_id: int):
        self.name = name
        self.classifier_id = classifier_id


class TrainingDocument(Base):
    __tablename__ = 'training_document'
    id = Column(Integer(), primary_key=True)
    text = Column(Text())
    md5 = Column(Text(), unique=True)
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True),
                          default=datetime.datetime.now,
                          onupdate=datetime.datetime.now)

    def __init__(self, text: str):
        self.text = text
        self.md5 = md5(text.encode('utf-8')).hexdigest()


class Classification(Base):
    __tablename__ = 'classification'
    id = Column(Integer(), primary_key=True)
    document_id = Column(Integer(),
                         ForeignKey('training_document.id', onupdate='CASCADE',
                                    ondelete='CASCADE'))
    document = relationship('TrainingDocument',
                            backref=backref('classifications',
                                            cascade="all, delete-orphan",
                                            passive_deletes=True))
    label_id = Column(Integer(), ForeignKey('label.id', onupdate='CASCADE',
                                            ondelete='CASCADE'))
    label = relationship('Label', backref=backref('classifications',
                                                  cascade="all, delete-orphan",
                                                  passive_deletes=True))
    classifier_id = Column(Integer(),
                           ForeignKey('classifier.id', onupdate='CASCADE',
                                      ondelete='CASCADE'))
    classifier = relationship('Classifier', backref=backref('classifications',
                                                            cascade="all, delete-orphan",
                                                            passive_deletes=True))
    user_id = Column(Integer(),
                        ForeignKey('user.id', onupdate='CASCADE',
                                   ondelete='RESTRICT'))
    user = relationship('User', backref=backref('classifications'))
    assigned = Column(Boolean())
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True),
                          default=datetime.datetime.now,
                          onupdate=datetime.datetime.now)
    __table_args__ = (UniqueConstraint('document_id', 'label_id', 'user_id'),)


class Job(Base):
    __tablename__ = 'job'
    id = Column(Integer(), primary_key=True)
    description = Column(Text())
    action = deferred(Column(PickleType()))
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    start = Column(DateTime(timezone=True))
    completion = Column(DateTime(timezone=True))
    status_pending = 'pending'
    status_done = 'done'
    status_error = 'error'
    status_running = 'running'
    status_missing = 'missing'

    status = Column(String(10), default=status_pending)

    def __init__(self, description: str, action):
        self.description = description
        self.action = action


class ClassificationJob(Base):
    __tablename__ = 'classificationjob'
    id = Column(Integer(), primary_key=True)
    dataset_id = Column(Integer(), ForeignKey('dataset.id', onupdate='CASCADE',
                                              ondelete='CASCADE'))
    dataset = relationship('Dataset', backref=backref('classifications',
                                                      cascade="all, delete-orphan",
                                                      passive_deletes=True))
    job_id = Column(Integer(), ForeignKey('job.id'))
    job = relationship('Job', backref=backref('classification_job'))
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    classifiers = Column(Text())
    filename = Column(Text())

    __mapper_args__ = {
        'polymorphic_identity': 'classification',
    }

    def __init__(self, dataset_id: int, classifiers: str, job_id: int, filename: str):
        self.dataset_id = dataset_id
        self.classifiers = classifiers
        self.job_id = job_id
        self.filename = filename


class Lock(Base):
    __tablename__ = 'lock'
    name = Column(String(classifier_name_length + 20), primary_key=True)
    locker = Column(String(40))
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)

    def __init__(self, name: str, locker: str):
        self.name = name
        self.locker = locker


class SQLAlchemyDB(object):
    def __init__(self, name: str, poolclass=None):
        self._engine = create_engine(name, poolclass=poolclass)
        Base.metadata.create_all(self._engine)
        self._sessionmaker = scoped_session(sessionmaker(bind=self._engine))
        configure_mappers()
        self._preload_data()

    def _preload_data(self):
        with self.session_scope() as session:
            if not session.query(
                    exists().where(User.name == _ADMIN_NAME)).scalar():
                session.add(User(_ADMIN_NAME, _ADMIN_PASSWORD, _NO_HOURLY_LIMIT,
                                 _NO_REQUEST_LIMIT))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._sessionmaker.session_factory.close_all()
        self._engine.dispose()

    @contextmanager
    def session_scope(self):
        session = self._sessionmaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def admin_name():
        return _ADMIN_NAME

    def create_user(self, name: str, password: str):
        with self.session_scope() as session:
            user = User(name, password)
            session.add(user)

    def user_exists(self, name: str):
        with self.session_scope() as session:
            return session.query(exists().where(User.name == name)).scalar()

    def verify_user(self, name: str, password: str):
        with self.session_scope() as session:
            user = session.query(User).filter(User.name == name).scalar()
            try:
                return user.verify(password)
            except:
                return False

    def delete_user(self, name: str):
        with self.session_scope() as session:
            user = session.query(User).filter(User.name == name).scalar()
            if user is not None:
                session.delete(user)

    def change_password(self, name: str, password: str):
        with self.session_scope() as session:
            user = session.query(User).filter(User.name == name).scalar()
            user.change_password(password)

    def user_names(self):
        with self.session_scope() as session:
            return self._flatten_list(
                session.query(User.name).order_by(User.name).all())

    def get_user_creation_time(self, name: str):
        with self.session_scope() as session:
            return session.query(User.creation).filter(
                User.name == name).scalar()

    def get_user_last_update_time(self, name: str):
        with self.session_scope() as session:
            return session.query(User.last_updated).filter(
                User.name == name).scalar()

    def get_user_hourly_limit(self, name: str):
        with self.session_scope() as session:
            return session.query(User.hourly_limit).filter(
                User.name == name).scalar()

    def get_user_request_limit(self, name: str):
        with self.session_scope() as session:
            return session.query(User.request_limit).filter(
                User.name == name).scalar()

    def get_user_current_request_counter(self, name: str):
        with self.session_scope() as session:
            user = session.query(User).filter(User.name == name).scalar()
            return user.check_current_request_counter()

    def get_user_total_request_counter(self, name: str):
        with self.session_scope() as session:
            return session.query(User.total_request_counter).filter(
                User.name == name).scalar()

    def user_check_and_count_request(self, name: str, cost: int):
        with self.session_scope() as session:
            key_obj = session.query(User).filter(User.name == name).scalar()
            if key_obj is None:
                return False
            return key_obj.check_and_count_request(cost)

    def set_user_hourly_limit(self, name: str, hourly_limit: int):
        with self.session_scope() as session:
            key_obj = session.query(User).filter(User.name == name).scalar()
            key_obj.hourly_limit = hourly_limit

    def set_user_request_limit(self, name: str, request_limit: int):
        with self.session_scope() as session:
            key_obj = session.query(User).filter(User.name == name).scalar()
            key_obj.request_limit = request_limit

    def set_user_current_request_counter(self, name: str, count: int):
        with self.session_scope() as session:
            key_obj = session.query(User).filter(User.name == name).scalar()
            key_obj.current_request_counter = count

    def classifier_names(self, only_public_classifiers=False):
        if only_public_classifiers:
            with self.session_scope() as session:
                return self._flatten_list(session.query(Classifier.name)
                                          .where(Classifier.public)
                                          .order_by(Classifier.name)
                                          .all())
        else:
            with self.session_scope() as session:
                return self._flatten_list(session.query(Classifier.name)
                                          .order_by(Classifier.name)
                                          .all())

    def classifier_exists(self, name: str):
        with self.session_scope() as session:
            return session.query(
                exists().where(Classifier.name == name)).scalar()

    def create_classifier(self, name: str, labels, classification_mode: ClassificationMode):
        with self.session_scope() as session:
            classifier = Classifier(name, classification_mode)
            session.add(classifier)
            session.flush()
            for label in labels:
                label_obj = Label(label, classifier.id)
                session.add(label_obj)

    def classifier_is_public(self, name):
        with self.session_scope() as session:
            classifier = session.query(Classifier).filter(Classifier.name == name).scalar()
            return classifier.public

    def classifier_set_public(self, name, public: bool = None):
        with self.session_scope() as session:
            classifier = session.query(Classifier).filter(Classifier.name == name).scalar()
            if public is None:
                classifier.public = not classifier.public
            else:
                classifier.public = public

    def add_classifier_label(self, classifier_name: str, label_name: str):
        with self.session_scope() as session:
            classifier = session.query(Classifier).filter(Classifier.name == classifier_name).scalar()
            label = (session.query(Label)
                     .filter(Label.name == label_name)
                     .join(Label.classifier)
                     .filter(Classifier.id == classifier.id)
                     .scalar()
                     )
            if label is not None:
                return
            label = Label(label_name, classifier.id)
            session.add(label)
        with self._lock_classifier_model(classifier_name):
            clf = self.get_classifier_model(classifier_name)
            if clf is not None:
                clf.add_label(label_name)
                self.update_classifier_model(classifier_name, clf)

    def get_classifier_model(self, name: str):
        with self.session_scope() as session:
            classifier = session.query(Classifier).filter(
                Classifier.name == name).scalar()
            if classifier.model is None:
                labels = self._flatten_list(
                    session.query(Label.name)
                        .join(Label.classifier)
                        .filter(Classifier.name == name)
                        .order_by(Label.name))
                classifier.model = create_classifier_model(name, labels)
            return classifier.model

    def get_preferred_classification_mode(self, name: str):
        with self.session_scope() as session:
            return session.query(
                Classifier.preferred_classification_mode).filter(
                Classifier.name == name).scalar()

    def set_preferred_classification_mode(self, name: str, classification_mode: ClassificationMode):
        with self.session_scope() as session:
            classifier = session.query(Classifier).filter(
                Classifier.name == name).scalar()
            classifier.preferred_classification_mode = classification_mode

    def get_classifier_labels(self, name: str):
        with self.session_scope() as session:
            labels = self._flatten_list(session.query(Label.name).
                                        join(Label.classifier).
                                        filter(Classifier.name == name).
                                        order_by(Label.name)
                                        )
            return labels

    def get_classifier_creation_time(self, name: str):
        with self.session_scope() as session:
            return session.query(Classifier.creation).filter(
                Classifier.name == name).scalar()

    def get_classifier_last_update_time(self, name: str):
        with self.session_scope() as session:
            return session.query(Classifier.last_updated).filter(
                Classifier.name == name).scalar()

    def rename_classifier(self, name: str, new_name: str, overwrite: bool):
        with self.session_scope() as session:
            classifier = session.query(Classifier).filter(
                Classifier.name == name).scalar()
            if classifier is None:
                return
            if overwrite:
                old_classifier = session.query(Classifier).filter(
                    Classifier.name == new_name).scalar()
                if old_classifier is not None:
                    session.delete(old_classifier)
                    session.flush()
            classifier.name = new_name

    def get_classifier_description(self, name: str):
        with self.session_scope() as session:
            return session.query(Classifier.description).filter(
                Classifier.name == name).scalar()

    def set_classifier_description(self, name: str, description: str):
        with self.session_scope() as session:
            classifier = session.query(Classifier).filter(
                Classifier.name == name).scalar()
            if classifier is None:
                return
            classifier.description = description

    def delete_classifier(self, name: str):
        with self.session_scope() as session:
            classifier = session.query(Classifier).filter(
                Classifier.name == name).scalar()
            if classifier is not None:
                session.delete(classifier)

    def update_classifier_model(self, name: str, model):
        with self.session_scope() as session:
            try:
                classifier = session.query(Classifier).filter(
                    Classifier.name == name).scalar()
                classifier.model = model
            except (OperationalError, MemoryError) as e:
                session.rollback()
                raise e

    def create_training_examples(self, classifier_name: str, X, y, usernames):
        if len(X) == 0:
            return

        with self.session_scope() as session:
            text_set = list(set(X))
            stmt = insert(TrainingDocument.__table__).values(
                [{'text': text,
                  'md5': md5(text.encode('utf-8')).hexdigest()}
                 for text in text_set])
            stmt = \
                stmt.on_conflict_do_update(index_elements=['md5'],
                                           set_={
                                               'last_updated':
                                                   datetime.datetime.now()})
            stmt = stmt.returning(TrainingDocument.id)
            with self._lock_all_training_documents():
                text_set_ids = [id_[0] for id_ in session.execute(stmt)]

            text_id_map = dict()
            for text, text_set_id in zip(text_set, text_set_ids):
                text_id_map[text] = text_set_id

            classifier_id = session.query(Classifier.id).filter(
                Classifier.name == classifier_name).scalar()

            user_id_map = dict()
            for username in set(usernames):
                user_id_map[username] = session.query(User.id).filter(User.name == username).scalar()

            labeling = dict()

            if type(y[0]) == str:
                label_set = self.get_classifier_labels(classifier_name)

                label_id_map = dict()

                for label in label_set:
                    label_id = (session.query(Label.id)
                                .filter(
                        Label.classifier_id == classifier_id)
                                .filter(Label.name == label)
                                .scalar()
                                )
                    label_id_map[label] = label_id

                for text, assigned_label in zip(X,y):
                    text_id = text_id_map[text]
                    for label in label_set:
                        if label == assigned_label:
                            labeling[(text_id, label_id_map[label], user_id_map[username])] = True
                        else:
                            labeling[(text_id, label_id_map[label], user_id_map[username])] = False

            else:
                label_set = set(label for (label, _) in y)

                label_id_map = dict()
                for label in label_set:
                    label_id = (session.query(Label.id)
                                .filter(
                        Label.classifier_id == classifier_id)
                                .filter(Label.name == label)
                                .scalar()
                                )
                    label_id_map[label] = label_id

                for text, (label, y_value), username in zip(X,y,usernames):
                    if type(y_value) is str:
                        y_value = y_value == YES_LABEL or y_value == 'True' or y_value == 'true' or y_value == '1' \
                                  or y_value == '+1' or y_value == 'yes' or y_value == 'YES' or y_value == 'y' \
                                  or y_value == 'Y' or y_value == 'TRUE' or y_value == 'T' or y_value == True
                    elif type(y_value) is int:
                        y_value = y_value > 0
                    labeling[(text_id_map[text], label_id_map[label], user_id_map[username])] = y_value

            values = [{'document_id': document_id,
                       'label_id': label_id,
                       'classifier_id': classifier_id,
                       'user_id': user_id,
                       'assigned': assigned} for
                      (document_id, label_id, user_id), assigned
                      in labeling.items()]

            stmt = insert(Classification.__table__).values(
                values)
            to_update = {c.name: c for c in stmt.excluded}
            del to_update['id']
            del to_update['creation']
            stmt = stmt.on_conflict_do_update(
                index_elements=['document_id', 'label_id', 'user_id'],
                set_=to_update)
            session.execute(stmt)
        return

    def classify(self, classifier_name: str, X, classification_mode: ClassificationMode = None,
                 show_gold_labels=True):
        if classification_mode is None:
            classification_mode = self.get_preferred_classification_mode(
                classifier_name)
        if len(self.get_classifier_labels(classifier_name)) < 1:
            if classification_mode == ClassificationMode.SINGLE_LABEL:
                return [None] * len(X)
            elif classification_mode == ClassificationMode.MULTI_LABEL:
                return [[]] * len(X)
        clf = self.get_classifier_model(classifier_name)
        y = list()
        if classification_mode == ClassificationMode.SINGLE_LABEL:
            if clf is not None:
                default_label = None
            else:
                default_label = self.get_classifier_labels(classifier_name)[0]
            for x in X:
                known_labels_assignment = self.get_labels_for_text(
                    classifier_name, x)
                human_label_found = False
                if known_labels_assignment is not None:
                    for label, assigned, user_name in known_labels_assignment:
                        if assigned:
                            y.append((label, show_gold_labels))
                            human_label_found = True
                            break
                if not human_label_found:
                    if clf is not None:
                        y.append((clf.classify([x], False)[0], False))
                    else:
                        y.append((default_label, False))
        elif classification_mode == ClassificationMode.MULTI_LABEL:
            for x in X:
                known_labels_assignment = self.get_labels_for_text(
                    classifier_name, x)
                known_labels = {}
                if known_labels_assignment is not None:
                    for label, assignment, user_name in known_labels_assignment:
                        if label not in known_labels:
                            if assignment:
                                known_labels[
                                    label] = (label, True, show_gold_labels)
                            else:
                                known_labels[
                                    label] = (label, False, show_gold_labels)
                pred_labels = {}
                if clf is not None:
                    label_values = clf.classify([x], True)[0]
                    for label, value in label_values:
                        pred_labels[label] = (label, value, False)
                y_x = list()
                for label in self.get_classifier_labels(classifier_name):
                    if label in known_labels:
                        y_x.append(known_labels[label])
                    elif label in pred_labels:
                        y_x.append(pred_labels[label])
                    else:
                        raise Exception(
                            f'Missing a classification: {classifier_name}, {label}')
                y.append(y_x)
        return y

    def score(self, classifier_name: str, X):
        clf = self.get_classifier_model(classifier_name)
        if clf is None:
            labels = self.get_classifier_labels(classifier_name)
            equi_prob = {label: 0 for label in labels}
            return [equi_prob for _ in X]
        scores = clf.decision_function(X)
        labels = clf.labels()
        return [dict(zip(labels, values)) for values in scores]

    def get_labels_for_text(self, classifier_name: str, text: str, by_last_update=True):
        if by_last_update:
            order_by = Classification.last_updated.desc()
        else:
            order_by = Label.name
        with self.session_scope() as session:
            return (session.query(Label.name, Classification.assigned, User.name)
                    .filter(TrainingDocument.md5 == md5(
                text.encode('utf-8')).hexdigest())
                    .join(Classification.document)
                    .join(Classification.classifier)
                    .join(Classification.label)
                    .join(Classification.user)
                    .filter(Classifier.name == classifier_name)
                    .order_by(order_by)
                    )

    def get_labels_of_training_id(self, classifier_name: str, trainingdocument_id: int, by_last_update=True):
        if by_last_update:
            order_by = Classification.last_updated.desc()
        else:
            order_by = Label.name
        with self.session_scope() as session:
            return (session.query(Label.name, Classification.assigned, User.name)
                    .join(Classification.document)
                    .join(Classification.classifier)
                    .join(Classification.label)
                    .join(Classification.user)
                    .filter(TrainingDocument.id == trainingdocument_id)
                    .filter(Classifier.name == classifier_name)
                    .order_by(order_by)
                    )

    def get_training_id_last_update_time(self, classifier_name: str, trainingdocument_id: int):
        with self.session_scope() as session:
            return (session.query(Classification.last_updated)
                    .join(Classification.document)
                    .join(Classification.classifier)
                    .filter(Classifier.name == classifier_name)
                    .filter(TrainingDocument.id == trainingdocument_id)
                    .order_by(Classification.last_updated.desc())
                    .first())[0]

    def rename_classifier_label(self, classifier_name: str, label_name: str, new_name: str):
        with self.session_scope() as session:
            label = (session.query(Label)
                     .filter(Label.name == label_name)
                     .join(Label.classifier)
                     .filter(Classifier.name == classifier_name)
                     .scalar()
                     )
            if label is None:
                return
            label.name = new_name
        with self._lock_classifier_model(classifier_name):
            clf = self.get_classifier_model(classifier_name)
            if clf is not None:
                clf.rename_label(label_name, new_name)
                self.update_classifier_model(classifier_name, clf)

    def delete_classifier_label(self, classifier_name: str, label_name: str):
        with self.session_scope() as session:
            label = (session.query(Label)
                     .filter(Label.name == label_name)
                     .join(Label.classifier)
                     .filter(Classifier.name == classifier_name)
                     .scalar()
                     )
            if label is None:
                return
            session.delete(label)
        with self._lock_classifier_model(classifier_name):
            clf = self.get_classifier_model(classifier_name)
            if clf is not None:
                clf.delete_label(label_name)
                self.update_classifier_model(classifier_name, clf)

    def dataset_names(self):
        with self.session_scope() as session:
            return self._flatten_list(
                session.query(Dataset.name).order_by(Dataset.name))

    def dataset_exists(self, name: str):
        with self.session_scope() as session:
            return session.query(exists().where(Dataset.name == name)).scalar()

    def create_dataset(self, name: str):
        with self.session_scope() as session:
            dataset = Dataset(name)
            session.add(dataset)

    def get_dataset_description(self, name: str):
        with self.session_scope() as session:
            return session.query(Dataset.description).filter(
                Dataset.name == name).scalar()

    def set_dataset_description(self, name: str, description: str):
        with self.session_scope() as session:
            dataset = session.query(Dataset).filter(
                Dataset.name == name).scalar()
            if dataset is None:
                return
            dataset.description = description

    def rename_dataset(self, name: str, newname: str):
        with self.session_scope() as session:
            dataset = session.query(Dataset).filter(
                Dataset.name == name).scalar()
            if dataset is None:
                return
            dataset.name = newname

    def delete_dataset(self, name: str):
        with self.session_scope() as session:
            dataset = session.query(Dataset).filter(
                Dataset.name == name).scalar()
            if dataset is not None:
                session.delete(dataset)

    def get_dataset_creation_time(self, name: str):
        with self.session_scope() as session:
            return session.query(Dataset.creation).filter(
                Dataset.name == name).scalar()

    def get_dataset_last_update_time(self, name: str):
        with self.session_scope() as session:
            return session.query(Dataset.last_updated).filter(
                Dataset.name == name).scalar()

    def get_dataset_size(self, name: str):
        with self.session_scope() as session:
            return session.query(Dataset.documents).filter(
                Dataset.name == name).filter(
                DatasetDocument.dataset_id == Dataset.id).count()

    def create_dataset_documents(self, dataset_name: str, external_ids_and_contents):
        with self.session_scope() as session:
            dataset = session.query(Dataset).filter(
                Dataset.name == dataset_name).one()
            values = dict()
            for external_id, content in external_ids_and_contents:
                values[external_id] = {'text': content,
                                       'md5': md5(
                                           content.encode('utf-8')).hexdigest(),
                                       'dataset_id': dataset.id,
                                       'external_id': external_id}
            stmt = insert(DatasetDocument.__table__).values(
                list(values.values()))
            stmt = stmt.on_conflict_do_update(
                index_elements=['dataset_id', 'external_id'],
                set_=dict(stmt.excluded))
            session.execute(stmt)
            dataset.last_updated = datetime.datetime.now()

    def delete_dataset_document(self, dataset_name: str, external_id: int):
        with self.session_scope() as session:
            dataset = session.query(Dataset).filter(
                Dataset.name == dataset_name).one()
            document = (session.query(DatasetDocument)
                        .filter(DatasetDocument.dataset_id == dataset.id)
                        .filter(DatasetDocument.external_id == external_id)
                        .scalar()
                        )
            if document is not None:
                session.delete(document)
            dataset.last_updated = datetime.datetime.now()

    def get_training_document_count(self, name: str):
        with self.session_scope() as session:
            count = (session.query(TrainingDocument.id)
                     .join(Classification.document)
                     .join(Classification.classifier)
                     .filter(Classifier.name == name)
                     .distinct()
                     .count()
                     )
            if count:
                return count
            else:
                return 0

    def get_classifier_examples_with_label_count(self, classifier_name: str, label: str):
        with self.session_scope() as session:
            count = (session.query(Classification.id)
                     .filter(Classifier.name == classifier_name)
                     .join(Classification.classifier)
                     .join(Classification.label)
                     .filter(Label.name == label)
                     .count())
        if count:
            return count
        else:
            return 0

    def delete_training_example(self, classifier_name: str, training_document_id: int):
        with self.session_scope() as session:
            classifications = (session.query(Classification)
                               .join(Classifier)
                               .join(TrainingDocument)
                               .filter(Classifier.name == classifier_name)
                               .filter(TrainingDocument.id == training_document_id)
                               .all())
            for classification in classifications:
                session.delete(classification)

    def get_training_documents(self, name: str, offset=0, limit: int = None, filter: str = None, by_last_update=False):
        if by_last_update:
            order_by = TrainingDocument.last_updated.desc()
        else:
            order_by = TrainingDocument.id
        if filter is None:
            filter = ''
        with self.session_scope() as session:
            return (session.query(TrainingDocument.text, TrainingDocument.id)
                    .filter(TrainingDocument.text.like(f'%{filter}%'))
                    .join(Classification.document)
                    .join(Classification.classifier)
                    .filter(Classifier.name == name)
                    .group_by(TrainingDocument)
                    .order_by(order_by)
                    .offset(offset)
                    .limit(limit)
                    )

    def get_classifier_examples_with_label(self, name: str, label: str, offset=0,
                                           limit: int = None):
        with self.session_scope() as session:
            return (session.query(Classification)
                    .filter(Classifier.name == name)
                    .join(Classification.classifier)
                    .join(Classification.label)
                    .filter(Label.name == label)
                    .order_by(Classification.id)
                    .offset(offset)
                    .limit(limit)
                    )

    def get_classifier_random_examples_with_label(self, name: str, label: str, limit: int = None):
        with self.session_scope() as session:
            return (session.query(Classification)
                    .filter(Classifier.name == name)
                    .join(Classification.classifier)
                    .join(Classification.label)
                    .filter(Label.name == label)
                    .order_by(func.random())
                    .limit(limit)
                    )

    def get_dataset_random_documents_without_labels(self, dataset_name: str,
                                                    classifier_name: str, filter: str = None,
                                                    limit: int = None,
                                                    require_all_labels=True):
        if filter is None:
            filter = ''
        with self.session_scope() as session:
            missing = list(
                session.query(DatasetDocument.text, DatasetDocument.id)
                    .join(DatasetDocument.dataset)
                    .filter(Dataset.name == dataset_name)
                    .filter(DatasetDocument.text.like('%' + filter + '%'))
                    .filter(not_(exists().where(
                    and_(DatasetDocument.md5 == TrainingDocument.md5,
                         Classification.classifier_id == Classifier.id,
                         Classifier.name == classifier_name,
                         TrainingDocument.id == Classification.document_id))))
                    .order_by(func.random())
                    .limit(limit)
            )
            if len(missing) > 0 or not require_all_labels:
                return missing

            labels = self.get_classifier_labels(classifier_name)
            partial = list(session.query(DatasetDocument.text, DatasetDocument.id)
                    .join(DatasetDocument.dataset)
                    .join(Classification.classifier)
                    .join(Classification.document)
                    .join(Classification.label)
                    .filter(Dataset.name == dataset_name)
                    .filter(DatasetDocument.text.like('%' + filter + '%'))
                    .filter(DatasetDocument.md5 == TrainingDocument.md5)
                    .filter(Classifier.name == classifier_name)
                    .filter(Label.name.in_(labels))
                    .group_by(DatasetDocument.id)
                    .having(count(Classification.id) != len(labels))
                    .order_by(func.random())
                    .limit(limit)
            )
            return partial

    def get_dataset_documents_without_labels_count(self, dataset_name: str, classifier_name: str,
                                                   require_all_labels=True):
        if not require_all_labels:
            with self.session_scope() as session:
                return (session.query(DatasetDocument.id)
                        .join(DatasetDocument.dataset)
                        .filter(Dataset.name == dataset_name)
                        .filter(not_(exists().where(
                    and_(DatasetDocument.md5 == TrainingDocument.md5,
                         Classification.classifier_id == Classifier.id,
                         Classifier.name == classifier_name,
                         TrainingDocument.id == Classification.document_id))))
                        .count()
                        )

        else:
            labels = self.get_classifier_labels(classifier_name)
            with self.session_scope() as session:
                count_partial = (session.query(DatasetDocument.id)
                                 .join(DatasetDocument.dataset)
                                 .filter(Dataset.name == dataset_name)
                                 .filter(
                    DatasetDocument.md5 == TrainingDocument.md5)
                                 .filter(
                    TrainingDocument.id == Classification.document_id)
                                 .filter(
                    Classification.classifier_id == Classifier.id)
                                 .filter(Classifier.name == classifier_name)
                                 .filter(Classification.label_id == Label.id)
                                 .filter(Label.name.in_(labels))
                                 .group_by(DatasetDocument.id)
                                 .having(
                    count(Classification.id) != len(labels))
                                 .count()
                                 )
                count_missing = (session.query(DatasetDocument.id)
                                 .join(DatasetDocument.dataset)
                                 .filter(Dataset.name == dataset_name)
                                 .filter(not_(exists().where(
                    and_(DatasetDocument.md5 == TrainingDocument.md5,
                         Classification.classifier_id == Classifier.id,
                         Classifier.name == classifier_name,
                         TrainingDocument.id == Classification.document_id))))
                                 .count()
                                 )
                return count_partial + count_missing

    def get_dataset_random_documents(self, dataset_name: str, filter: str = None, limit: int = None):
        if filter is None:
            filter = ''
        with self.session_scope() as session:
            return (session.query(DatasetDocument)
                    .join(DatasetDocument.dataset)
                    .filter(Dataset.name == dataset_name)
                    .filter(DatasetDocument.text.like('%' + filter + '%'))
                    .order_by(func.random())
                    .limit(limit)
                    )

    def get_dataset_next_documents(self, name: str, start_from: int, filter: str = None, limit: int = None):
        if filter is None:
            filter = ''

        start_document_id = self.get_dataset_document_by_position(name, start_from).id

        with self.session_scope() as session:
            return (session.query(DatasetDocument)
                    .filter(DatasetDocument.id >= start_document_id)
                    .filter(DatasetDocument.text.like('%' + filter + '%'))
                    .filter(Dataset.name == name)
                    .join(DatasetDocument.dataset)
                    .order_by(DatasetDocument.id)
                    .limit(limit)
                    )

    def get_dataset_next_documents_without_labels(self, dataset_name: str, classifier_name: str, start_from: int,
                                                  filter: str = None, limit: int = None, require_all_labels=True):
        if filter is None:
            filter = ''

        start_document_id = self.get_dataset_document_by_position(dataset_name, start_from).id

        with self.session_scope() as session:
            missing = list(
                session.query(DatasetDocument.text, DatasetDocument.id)
                    .join(DatasetDocument.dataset)
                    .filter(Dataset.name == dataset_name)
                    .filter(DatasetDocument.id >= start_document_id)
                    .filter(DatasetDocument.text.like('%' + filter + '%'))
                    .filter(not_(exists().where(
                    and_(DatasetDocument.md5 == TrainingDocument.md5,
                         Classification.classifier_id == Classifier.id,
                         Classifier.name == classifier_name,
                         TrainingDocument.id == Classification.document_id))))
                    .order_by(DatasetDocument.id)
                    .limit(limit)
            )
            if len(missing) > 0 or not require_all_labels:
                return missing

            labels = self.get_classifier_labels(classifier_name)
            partial = list(session.query(DatasetDocument.text, DatasetDocument.id)
                    .join(DatasetDocument.dataset)
                    .join(Classification.classifier)
                    .join(Classification.document)
                    .join(Classification.label)
                    .filter(Dataset.name == dataset_name)
                    .filter(DatasetDocument.text.like('%' + filter + '%'))
                    .filter(DatasetDocument.md5 == TrainingDocument.md5)
                    .filter(Classifier.name == classifier_name)
                    .filter(Label.name.in_(labels))
                    .group_by(DatasetDocument.id)
                    .having(count(Classification.id) != len(labels))
                    .order_by(DatasetDocument.id)
                    .limit(limit)
            )
            return partial

    def get_dataset_prev_documents(self, name: str, start_from: int, filter: str = None, limit: int = None):
        if filter is None:
            filter = ''

        start_document = self.get_dataset_document_by_position(name, start_from)

        with self.session_scope() as session:
            return (session.query(DatasetDocument)
                    .filter(DatasetDocument.id <= start_document.id)
                    .filter(DatasetDocument.text.like('%' + filter + '%'))
                    .filter(Dataset.name == name)
                    .join(DatasetDocument.dataset)
                    .order_by(DatasetDocument.id.desc())
                    .limit(limit)
                    )

    def get_dataset_prev_documents_without_labels(self, dataset_name: str, classifier_name: str, start_from: int,
                                                  filter: str = None, limit: int = None, require_all_labels=True):
        if filter is None:
            filter = ''

        start_document_id = self.get_dataset_document_by_position(dataset_name, start_from).id

        with self.session_scope() as session:
            missing = list(
                session.query(DatasetDocument.text, DatasetDocument.id)
                    .join(DatasetDocument.dataset)
                    .filter(Dataset.name == dataset_name)
                    .filter(DatasetDocument.id <= start_document_id)
                    .filter(DatasetDocument.text.like('%' + filter + '%'))
                    .filter(not_(exists().where(
                    and_(DatasetDocument.md5 == TrainingDocument.md5,
                         Classification.classifier_id == Classifier.id,
                         Classifier.name == classifier_name,
                         TrainingDocument.id == Classification.document_id))))
                    .order_by(DatasetDocument.id.desc())
                    .limit(limit)
            )
            if len(missing) > 0 or not require_all_labels:
                return missing

            labels = self.get_classifier_labels(classifier_name)
            partial = list(session.query(DatasetDocument.text, DatasetDocument.id)
                    .join(DatasetDocument.dataset)
                    .join(Classification.classifier)
                    .join(Classification.document)
                    .join(Classification.label)
                    .filter(Dataset.name == dataset_name)
                    .filter(DatasetDocument.text.like('%' + filter + '%'))
                    .filter(DatasetDocument.md5 == TrainingDocument.md5)
                    .filter(Classifier.name == classifier_name)
                    .filter(Label.name.in_(labels))
                    .group_by(DatasetDocument.id)
                    .having(count(Classification.id) != len(labels))
                    .order_by(DatasetDocument.id.desc())
                    .limit(limit)
            )
            return partial

    def get_dataset_documents(self, name: str, filter: str = None, offset=0, limit: int = None):
        if filter is None:
            filter = ''
        with self.session_scope() as session:
            return (session.query(DatasetDocument)
                    .filter(DatasetDocument.text.like('%' + filter + '%'))
                    .filter(Dataset.name == name)
                    .join(DatasetDocument.dataset)
                    .order_by(DatasetDocument.external_id)
                    .offset(offset)
                    .limit(limit)
                    )

    def get_dataset_document_by_name(self, datasetname: str, documentname: str):
        with self.session_scope() as session:
            document = (session.query(DatasetDocument)
                        .filter(Dataset.name == datasetname)
                        .join(DatasetDocument.dataset)
                        .filter(DatasetDocument.external_id == documentname)
                        .first()
                        )
            if document is not None:
                session.expunge(document)
            return document

    def get_dataset_document_by_position(self, name: str, position: int):
        with self.session_scope() as session:
            document = (session.query(DatasetDocument)
                        .filter(Dataset.name == name)
                        .join(DatasetDocument.dataset)
                        .order_by(DatasetDocument.id)
                        .offset(position)
                        .limit(1)
                        .scalar()
                        )
            if document is not None:
                session.expunge(document)
            return document

    def get_dataset_document_position_by_id(self, name: str, document_id: int):
        with self.session_scope() as session:
            return (session.query(DatasetDocument)
                    .filter(Dataset.name == name)
                    .join(DatasetDocument.dataset)
                    .filter(DatasetDocument.id < document_id)
                    .order_by(DatasetDocument.id)
                    .count()
                    )

    def get_jobs(self, starttime: int = None):
        if starttime is None:
            with self.session_scope() as session:
                return (session.query(Job)
                        .order_by(Job.creation.desc())
                        )
        else:
            with self.session_scope() as session:
                return (session.query(Job)
                        .filter(Job.creation > starttime)
                        .order_by(Job.creation.desc())
                        )

    def get_locks(self):
        with self.session_scope() as session:
            return session.query(Lock).order_by(Lock.creation.desc())

    def create_job(self, function, args=(), kwargs=None, description: str = None):
        if kwargs is None:
            kwargs = {}
        with self.session_scope() as session:
            if description is None:
                description = function.__name__
            job = Job(description,
                      {'function': function, 'args': args, 'kwargs': kwargs})
            session.add(job)
            session.commit()
            return job.id

    def get_next_pending_job(self):
        with self.session_scope() as session:
            job = (session.query(Job)
                   .filter(Job.status == Job.status_pending)
                   .order_by(Job.creation.asc())
                   .first()
                   )
            if job is None:
                return None
            job.action
            session.expunge(job)
            return job

    def set_job_completion_time(self, job_id: int, completion: int = None):
        if completion is None:
            completion = datetime.datetime.now()
        with self.session_scope() as session:
            job = session.query(Job).filter(Job.id == job_id).one()
            job.completion = completion

    def set_job_start_time(self, job_id: int, start: int = None):
        if start is None:
            start = datetime.datetime.now()
        with self.session_scope() as session:
            job = session.query(Job).filter(Job.id == job_id).one()
            job.start = start

    def set_job_status(self, job_id: int, status: str):
        with self.session_scope() as session:
            job = session.query(Job).filter(Job.id == job_id).one()
            job.status = status

    def get_job_status(self, job_id: int):
        with self.session_scope() as session:
            job = session.query(Job).filter(Job.id == job_id).first()
            if job is None:
                return Job.status_missing
            return job.status

    def get_most_recent_classifier_update_time(self, classifiers):
        with self.session_scope() as session:
            return (session.query(Classifier.last_updated)
                    .filter(Classifier.name.in_(classifiers))
                    .order_by(Classifier.last_updated.desc())
                    .first()
                    )[0]

    def create_classification_job(self, datasetname: str, classifiers, job_id: int, fullpath: str):
        with self.session_scope() as session:
            dataset_id = session.query(Dataset.id).filter(
                Dataset.name == datasetname).scalar()
            classification_job = ClassificationJob(dataset_id,
                                                   ', '.join(classifiers),
                                                   job_id, fullpath)
            session.add(classification_job)

    def get_classification_jobs(self, name: str):
        with self.session_scope() as session:
            return (session.query(ClassificationJob)
                    .join(ClassificationJob.dataset)
                    .filter(Dataset.name == name)
                    .order_by(ClassificationJob.creation)
                    )

    def get_classification_job_filename(self, id: int):
        with self.session_scope() as session:
            return (session.query(ClassificationJob.filename)
                    .filter(ClassificationJob.id == id)
                    .scalar()
                    )

    def delete_classification_job(self, id: int):
        with self.session_scope() as session:
            classification_job = session.query(ClassificationJob).filter(
                ClassificationJob.id == id).scalar()
            if os.path.exists(classification_job.filename):
                os.remove(classification_job.filename)
            if classification_job is not None:
                session.delete(classification_job)

    def delete_job(self, id: int):
        with self.session_scope() as session:
            job = session.query(Job).filter(Job.id == id).scalar()
            if job is not None:
                session.delete(job)

    def delete_lock(self, name: str):
        with self.session_scope() as session:
            lock = session.query(Lock).filter(Lock.name == name).scalar()
            if lock is not None:
                session.delete(lock)

    def classification_exists(self, filename: str):
        with self.session_scope() as session:
            return session.query(
                exists().where(ClassificationJob.filename == filename)).scalar()

    def job_exists(self, id: int):
        with self.session_scope() as session:
            return session.query(exists().where(Job.id == id)).scalar()

    def ipaddresses(self):
        with self.session_scope() as session:
            return self._flatten_list(
                session.query(IP.ip).order_by(IP.ip))

    def get_iptracker_creation_time(self, ip: str):
        with self.session_scope() as session:
            return session.query(IP.creation).filter(
                IP.ip == ip).scalar()

    def get_iptracker_last_update_time(self, ip: str):
        with self.session_scope() as session:
            return session.query(IP.last_updated).filter(
                IP.ip == ip).scalar()

    def get_iptracker_hourly_limit(self, ip: str):
        with self.session_scope() as session:
            return session.query(IP.hourly_limit).filter(
                IP.ip == ip).scalar()

    def set_iptracker_hourly_limit(self, ip: str, hourly_limit: int):
        with self.session_scope() as session:
            iptracker = session.query(IP).filter(
                IP.ip == ip).scalar()
            iptracker.hourly_limit = hourly_limit

    def get_iptracker_request_limit(self, ip: str):
        with self.session_scope() as session:
            return session.query(IP.request_limit).filter(
                IP.ip == ip).scalar()

    def set_iptracker_request_limit(self, ip: str, request_limit: int):
        with self.session_scope() as session:
            iptracker = session.query(IP).filter(
                IP.ip == ip).scalar()
            iptracker.request_limit = request_limit

    def get_iptracker_total_request_counter(self, ip: str):
        with self.session_scope() as session:
            return session.query(IP.total_request_counter).filter(
                IP.ip == ip).scalar()

    def get_iptracker_current_request_counter(self, ip: str):
        with self.session_scope() as session:
            iptracker = session.query(IP).filter(
                IP.ip == ip).scalar()
            return iptracker.check_current_request_counter()

    def set_iptracker_current_request_counter(self, ip: str, count: int):
        with self.session_scope() as session:
            iptracker = session.query(IP).filter(
                IP.ip == ip).scalar()
            iptracker.current_request_counter = count

    def iptracker_check_and_count_request(self, ip: str, cost: int):
        with self.session_scope() as session:
            iptracker = session.query(IP).filter(
                IP.ip == ip).scalar()
            if iptracker is None:
                raise LookupError()
            return iptracker.check_and_count_request(cost)

    def create_iptracker(self, ip: str, hourly_limit: int, request_limit: int):
        with self.session_scope() as session:
            iptracker = IP(ip, hourly_limit, request_limit)
            session.add(iptracker)

    def delete_iptracker(self, ip: str):
        with self.session_scope() as session:
            ip = session.query(IP).filter(IP.ip == ip).scalar()
            if ip is not None:
                session.delete(ip)

    def acquire_lock(self, name: str, locker: str, poll_interval=1):
            locked = False
            while not locked:
                with self.session_scope() as session:
                    try:
                        lock = Lock(name, locker)
                        session.add(lock)
                        session.commit()
                        locked = True
                    except:
                        session.rollback()
                        time.sleep(poll_interval)

    def release_lock(self, name: str, locker: str):
        with self.session_scope() as session:
            lock = session.query(Lock).filter(Lock.name == name).filter(
                Lock.locker == locker).first()
            if lock is not None:
                session.delete(lock)

    def keys(self):
        with self.session_scope() as session:
            return self._flatten_list(
                session.query(KeyTracker.key).order_by(KeyTracker.key))

    def get_keytracker_name(self, key: str):
        with self.session_scope() as session:
            return session.query(KeyTracker.name).filter(
                KeyTracker.key == key).scalar()

    def get_keytracker_creation_time(self, key: str):
        with self.session_scope() as session:
            return session.query(KeyTracker.creation).filter(
                KeyTracker.key == key).scalar()

    def get_keytracker_last_update_time(self, key: str):
        with self.session_scope() as session:
            return session.query(KeyTracker.last_updated).filter(
                KeyTracker.key == key).scalar()

    def get_keytracker_hourly_limit(self, key: str):
        with self.session_scope() as session:
            return session.query(KeyTracker.hourly_limit).filter(
                KeyTracker.key == key).scalar()

    def get_keytracker_request_limit(self, key: str):
        with self.session_scope() as session:
            return session.query(KeyTracker.request_limit).filter(
                KeyTracker.key == key).scalar()

    def get_keytracker_current_request_counter(self, key: str):
        with self.session_scope() as session:
            keytracker = session.query(KeyTracker).filter(
                KeyTracker.key == key).scalar()
            return keytracker.check_current_request_counter()

    def get_keytracker_total_request_counter(self, key: str):
        with self.session_scope() as session:
            return session.query(KeyTracker.total_request_counter).filter(
                KeyTracker.key == key).scalar()

    def keytracker_check_and_count_request(self, key: str, cost: int):
        with self.session_scope() as session:
            key_obj = session.query(KeyTracker).filter(
                KeyTracker.key == key).scalar()
            if key_obj is None:
                return False
            return key_obj.check_and_count_request(cost)

    def set_keytracker_hourly_limit(self, key: str, hourly_limit: int):
        with self.session_scope() as session:
            key_obj = session.query(KeyTracker).filter(
                KeyTracker.key == key).scalar()
            key_obj.hourly_limit = hourly_limit

    def set_keytracker_request_limit(self, key: str, request_limit: int):
        with self.session_scope() as session:
            key_obj = session.query(KeyTracker).filter(
                KeyTracker.key == key).scalar()
            key_obj.request_limit = request_limit

    def set_keytracker_current_request_counter(self, key: str, count: int):
        with self.session_scope() as session:
            key_obj = session.query(KeyTracker).filter(
                KeyTracker.key == key).scalar()
            key_obj.current_request_counter = count

    def create_keytracker(self, name: str, hourly_limit: int, request_limit: int):
        with self.session_scope() as session:
            key_obj = KeyTracker(name, hourly_limit, request_limit)
            session.add(key_obj)
            return key_obj.key

    def delete_keytracker(self, key: str):
        with self.session_scope() as session:
            key = session.query(KeyTracker).filter(
                KeyTracker.key == key).scalar()
            if key is not None:
                session.delete(key)

    def _flatten_list(self, list_of_list):
        return [item for sublist in list_of_list for item in sublist]

    def _lock_all_training_documents(self):
        return DBLock(self, 'all_training_documents')

    def _lock_classifier_training_documents(self, classifier_name: str):
        return DBLock(self, f'{classifier_name} training_documents')

    def _lock_classifier_model(self, classifier_name: str):
        return DBLock(self, f'{classifier_name} model')

    @staticmethod
    def version():
        import ics
        return ics.__version__


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

    def __del__(self):
        self.release()
        return None
