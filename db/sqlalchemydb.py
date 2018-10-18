import datetime
import os
import secrets
import time
from contextlib import contextmanager
from hashlib import md5
from uuid import uuid4

from passlib.hash import pbkdf2_sha256
from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Text, create_engine, PickleType, \
    UniqueConstraint, exists, not_, and_, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, deferred, relationship, configure_mappers, backref
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.sql.functions import count

from classifier.classifier import get_classifier_model, get_classifier_type_from_model, CLASSIFIER_TYPES, SINGLE_LABEL, \
    MULTI_LABEL
from classifier.multilabel_online_classifier import YES_LABEL, NO_LABEL

__author__ = 'Andrea Esuli'

Base = declarative_base()

_ADMIN_NAME = 'admin'
_ADMIN_PASSWORD = 'adminadmin'
_NO_HOURLY_LIMIT = -1
_NO_REQUEST_LIMIT = -1
_CLASSIFIER_LABEL_SEP = ' - '
_CLASSIFIER_TYPE_HIDDEN_FOR_MULTI_LABEL = 'Hidden'
_HUMAN_MADE = '✍'

classifier_name_length = 80
classifier_description_length = 300
user_name_length = 50
key_name_length = 50
ipaddress_length = 45
key_length = 30
label_name_length = 50
dataset_name_length = 100
document_name_length = 100
salt_length = 20


class Tracker(Base):
    __tablename__ = 'tracker'
    id = Column(Integer(), primary_key=True)
    hourly_limit = Column(Integer())
    current_request_counter = Column(Integer(), default=0)
    counter_time_span = Column(Integer(), default=int(time.time() / 3600))
    total_request_counter = Column(Integer(), default=0)
    request_limit = Column(Integer(), default=0)
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True), default=datetime.datetime.now)
    type = Column(String(20))

    __mapper_args__ = {
        'polymorphic_identity': 'tracker',
        'polymorphic_on': type
    }

    def __init__(self, hourly_limit, request_limit):
        self.hourly_limit = int(hourly_limit)
        self.request_limit = int(request_limit)

    def check_and_count_request(self, cost=1):
        if self.request_limit >= 0 and self.total_request_counter >= self.request_limit:
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
    id = Column(Integer, ForeignKey('tracker.id', onupdate='CASCADE', ondelete='CASCADE'), primary_key=True)
    key = Column(String(key_length * 2), unique=True)
    name = Column(String(key_name_length), unique=True)
    __mapper_args__ = {
        'polymorphic_identity': 'keytracker',
    }

    def __init__(self, name, hourly_limit, request_limit):
        super().__init__(hourly_limit, request_limit)
        self.key = secrets.token_hex(key_length)
        self.name = name


class IPTracker(Tracker):
    __tablename__ = 'ip'
    id = Column(Integer, ForeignKey('tracker.id', onupdate='CASCADE', ondelete='CASCADE'), primary_key=True)
    ip = Column(String(ipaddress_length), unique=True)
    __mapper_args__ = {
        'polymorphic_identity': 'iptracker',
    }

    def __init__(self, ip, hourly_limit, request_limit):
        super().__init__(hourly_limit, request_limit)
        self.ip = ip.strip()


class User(Tracker):
    __tablename__ = 'user'
    id = Column(Integer, ForeignKey('tracker.id', onupdate='CASCADE', ondelete='CASCADE'), primary_key=True)
    name = Column(String(user_name_length), unique=True)
    salted_password = Column(String())
    __mapper_args__ = {
        'polymorphic_identity': 'usertracker',
    }

    def __init__(self, name, password, hourly_limit=_NO_HOURLY_LIMIT,
                 request_limit=_NO_REQUEST_LIMIT):
        super().__init__(hourly_limit, request_limit)
        self.name = name
        self.salted_password = pbkdf2_sha256.hash(password)

    def verify(self, password):
        return pbkdf2_sha256.verify(password, self.salted_password)

    def change_password(self, password):
        self.salted_password = pbkdf2_sha256.hash(password)


class Dataset(Base):
    __tablename__ = 'dataset'
    id = Column(Integer(), primary_key=True)
    name = Column(String(dataset_name_length), unique=True)
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True), default=datetime.datetime.now, onupdate=datetime.datetime.now)

    def __init__(self, name):
        self.name = name


class DatasetDocument(Base):
    __tablename__ = 'dataset_document'
    id = Column(Integer(), primary_key=True, index=True)
    external_id = Column(String(document_name_length))
    dataset_id = Column(Integer(), ForeignKey('dataset.id', onupdate='CASCADE', ondelete='CASCADE'))
    dataset = relationship('Dataset', backref=backref('documents',
                                                      cascade="all, delete-orphan",
                                                      passive_deletes=True))
    text = Column(Text())
    md5 = Column(Text(), index=True)
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now, onupdate=datetime.datetime.now)
    __table_args__ = (UniqueConstraint('dataset_id', 'external_id'),)

    def __init__(self, text, dataset_id, external_id=None):
        self.text = text
        self.md5 = md5(text.encode('utf-8')).hexdigest()
        self.dataset_id = dataset_id
        self.external_id = external_id


class Classifier(Base):
    __tablename__ = 'classifier'
    id = Column(Integer(), primary_key=True)
    name = Column(String(classifier_name_length), unique=True)
    description = Column(String(classifier_description_length), default="No description")
    classifier_type = Column(String(max([len(name) + 2 for name in CLASSIFIER_TYPES])))
    model = deferred(Column(PickleType()))
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    last_updated = Column(DateTime(timezone=True), default=datetime.datetime.now, onupdate=datetime.datetime.now)

    def __init__(self, name, classifier_type):
        self.name = name
        self.model = None
        if not classifier_type in CLASSIFIER_TYPES and classifier_type != _CLASSIFIER_TYPE_HIDDEN_FOR_MULTI_LABEL:
            raise ValueError('Unknown classifier type')
        self.classifier_type = classifier_type


class Label(Base):
    __tablename__ = 'label'
    id = Column(Integer(), primary_key=True)
    name = Column(String(label_name_length), nullable=False)
    classifier_id = Column(Integer(), ForeignKey('classifier.id', onupdate='CASCADE', ondelete='CASCADE'))
    classifier = relationship('Classifier', backref=backref('labels',
                                                            cascade="all, delete-orphan",
                                                            passive_deletes=True))
    __table_args__ = (UniqueConstraint('classifier_id', 'name'),)

    def __init__(self, name, classifier_id):
        self.name = name
        self.classifier_id = classifier_id


class TrainingDocument(Base):
    __tablename__ = 'training_document'
    id = Column(Integer(), primary_key=True)
    text = Column(Text())
    md5 = Column(Text(), unique=True)
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now, onupdate=datetime.datetime.now)

    def __init__(self, text):
        self.text = text
        self.md5 = md5(text.encode('utf-8')).hexdigest()


class Classification(Base):
    __tablename__ = 'classification'
    id = Column(Integer(), primary_key=True)
    document_id = Column(Integer(), ForeignKey('training_document.id', onupdate='CASCADE', ondelete='CASCADE'))
    document = relationship('TrainingDocument', backref=backref('classifications',
                                                                cascade="all, delete-orphan",
                                                                passive_deletes=True))
    label_id = Column(Integer(), ForeignKey('label.id', onupdate='CASCADE', ondelete='CASCADE'))
    label = relationship('Label', backref=backref('classifications',
                                                  cascade="all, delete-orphan",
                                                  passive_deletes=True))
    classifier_id = Column(Integer(), ForeignKey('classifier.id', onupdate='CASCADE', ondelete='CASCADE'))
    classifier = relationship('Classifier', backref=backref('classifications',
                                                            cascade="all, delete-orphan",
                                                            passive_deletes=True))
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)
    __table_args__ = (UniqueConstraint('document_id', 'classifier_id'),)

    def __init__(self, document_id, classifier_id, label_id):
        self.document_id = document_id
        self.classifier_id = classifier_id
        self.label_id = label_id


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

    def __init__(self, description, action):
        self.description = description
        self.action = action


class ClassificationJob(Base):
    __tablename__ = 'classificationjob'
    id = Column(Integer(), primary_key=True)
    dataset_id = Column(Integer(), ForeignKey('dataset.id', onupdate='CASCADE', ondelete='CASCADE'))
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

    def __init__(self, dataset_id, classifiers, job_id, filename):
        self.dataset_id = dataset_id
        self.classifiers = classifiers
        self.job_id = job_id
        self.filename = filename


class Lock(Base):
    __tablename__ = 'lock'
    name = Column(String(classifier_name_length + 20), primary_key=True)
    locker = Column(String(40))
    creation = Column(DateTime(timezone=True), default=datetime.datetime.now)

    def __init__(self, name, locker):
        self.name = name
        self.locker = locker


class SQLAlchemyDB(object):
    def __init__(self, name):
        self._engine = create_engine(name)
        Base.metadata.create_all(self._engine)
        self._sessionmaker = scoped_session(sessionmaker(bind=self._engine))
        configure_mappers()
        self._preload_data()

    def _preload_data(self):
        with self.session_scope() as session:
            if not session.query(exists().where(User.name == _ADMIN_NAME)).scalar():
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
        return _ADMIN_NAME

    def create_user(self, name, password):
        with self.session_scope() as session:
            user = User(name, password)
            session.add(user)

    def user_exists(self, name):
        with self.session_scope() as session:
            return session.query(exists().where(User.name == name)).scalar()

    def verify_user(self, name, password):
        with self.session_scope() as session:
            user = session.query(User).filter(User.name == name).scalar()
            try:
                return user.verify(password)
            except:
                return False

    def delete_user(self, name):
        with self.session_scope() as session:
            user = session.query(User).filter(User.name == name).scalar()
            if user is not None:
                session.delete(user)

    def change_password(self, name, password):
        with self.session_scope() as session:
            user = session.query(User).filter(User.name == name).scalar()
            user.change_password(password)

    def user_names(self):
        with self.session_scope() as session:
            return self._flatten_list(session.query(User.name).order_by(User.name).all())

    def get_user_creation_time(self, name):
        with self.session_scope() as session:
            return session.query(User.creation).filter(User.name == name).scalar()

    def get_user_last_update_time(self, name):
        with self.session_scope() as session:
            return session.query(User.last_updated).filter(User.name == name).scalar()

    def get_user_hourly_limit(self, name):
        with self.session_scope() as session:
            return session.query(User.hourly_limit).filter(User.name == name).scalar()

    def get_user_request_limit(self, name):
        with self.session_scope() as session:
            return session.query(User.request_limit).filter(User.name == name).scalar()

    def get_user_current_request_counter(self, name):
        with self.session_scope() as session:
            user = session.query(User).filter(User.name == name).scalar()
            return user.check_current_request_counter()

    def get_user_total_request_counter(self, name):
        with self.session_scope() as session:
            return session.query(User.total_request_counter).filter(User.name == name).scalar()

    def user_check_and_count_request(self, name, cost):
        with self.session_scope() as session:
            key_obj = session.query(User).filter(User.name == name).scalar()
            if key_obj is None:
                return False
            return key_obj.check_and_count_request(cost)

    def set_user_hourly_limit(self, name, hourly_limit):
        with self.session_scope() as session:
            key_obj = session.query(User).filter(User.name == name).scalar()
            key_obj.hourly_limit = hourly_limit

    def set_user_request_limit(self, name, request_limit):
        with self.session_scope() as session:
            key_obj = session.query(User).filter(User.name == name).scalar()
            key_obj.request_limit = request_limit

    def set_user_current_request_counter(self, name, count):
        with self.session_scope() as session:
            key_obj = session.query(User).filter(User.name == name).scalar()
            key_obj.current_request_counter = count

    def classifier_names(self, include_hidden=False):
        if include_hidden:
            with self.session_scope() as session:
                return self._flatten_list(session.query(Classifier.name)
                                          .order_by(Classifier.name)
                                          .all())
        else:
            with self.session_scope() as session:
                return self._flatten_list(session.query(Classifier.name)
                                          .filter(Classifier.classifier_type != _CLASSIFIER_TYPE_HIDDEN_FOR_MULTI_LABEL)
                                          .order_by(Classifier.name)
                                          .all())

    def classifier_exists(self, name):
        with self.session_scope() as session:
            return session.query(exists().where(Classifier.name == name)).scalar()

    def create_classifier(self, name, labels, classifier_type):
        with self.session_scope() as session:
            if classifier_type == SINGLE_LABEL:
                classifier = Classifier(name, classifier_type)
                session.add(classifier)
                session.flush()
                for label in labels:
                    label_obj = Label(label, classifier.id)
                    session.add(label_obj)
            elif classifier_type == MULTI_LABEL:
                classifier = Classifier(name, classifier_type)
                session.add(classifier)
                session.flush()
                main_classifier_id = classifier.id
                for label in labels:
                    label_obj = Label(label, main_classifier_id)
                    session.add(label_obj)
                    classifier = Classifier(name + _CLASSIFIER_LABEL_SEP + label,
                                            _CLASSIFIER_TYPE_HIDDEN_FOR_MULTI_LABEL)
                    session.add(classifier)
                    session.flush()
                    label_obj = Label(YES_LABEL, classifier.id)
                    session.add(label_obj)
                    label_obj = Label(NO_LABEL, classifier.id)
                    session.add(label_obj)

    def get_classifier_model(self, name):
        with self.session_scope() as session:
            classifier = session.query(Classifier).filter(Classifier.name == name).scalar()
            if classifier.model is None:
                labels = self._flatten_list(
                    session.query(Label.name)
                        .join(Label.classifier)
                        .filter(Classifier.name == name)
                        .order_by(Label.name))
                classifier.model = get_classifier_model(classifier.classifier_type, name, labels)
            return classifier.model

    def get_classifier_type(self, name):
        with self.session_scope() as session:
            return session.query(Classifier.classifier_type).filter(Classifier.name == name).scalar()

    def get_classifier_labels(self, name):
        with self.session_scope() as session:
            labels = self._flatten_list(session.query(Label.name).
                                        join(Label.classifier).
                                        filter(Classifier.name == name).
                                        order_by(Label.name)
                                        )
            return labels

    def get_classifier_creation_time(self, name):
        with self.session_scope() as session:
            return session.query(Classifier.creation).filter(Classifier.name == name).scalar()

    def get_classifier_last_update_time(self, name):
        with self.session_scope() as session:
            return session.query(Classifier.last_updated).filter(Classifier.name == name).scalar()

    def rename_classifier(self, name, newname, overwrite):
        with self.session_scope() as session:
            classifier = session.query(Classifier).filter(Classifier.name == name).scalar()
            if classifier is None:
                return
            if overwrite:
                try:
                    old_classifier_type = self.get_classifier_type(newname)
                except:
                    old_classifier_type = None
                if old_classifier_type == SINGLE_LABEL:
                    old_classifier = session.query(Classifier).filter(Classifier.name == newname).scalar()
                    session.delete(old_classifier)
                    session.flush()
                elif old_classifier_type == MULTI_LABEL:
                    labels = self.get_classifier_labels(newname)
                    for label in labels:
                        old_classifier = session.query(Classifier).filter(
                            Classifier.name == newname + _CLASSIFIER_LABEL_SEP + label).scalar()
                        session.delete(old_classifier)
                        session.flush()
                    old_classifier = session.query(Classifier).filter(Classifier.name == newname).scalar()
                    session.delete(old_classifier)
                    session.flush()
            classifier.name = newname
            session.flush()
            labels = self.get_classifier_labels(newname)
            for label in labels:
                classifier = session.query(Classifier).filter(
                    Classifier.name == name + _CLASSIFIER_LABEL_SEP + label).scalar()
                classifier.name = newname + _CLASSIFIER_LABEL_SEP + label

    def get_classifier_description(self, name):
        with self.session_scope() as session:
            return session.query(Classifier.description).filter(Classifier.name == name).scalar()

    def set_classifier_description(self, name, description):
        with self.session_scope() as session:
            classifier = session.query(Classifier).filter(Classifier.name == name).scalar()
            if classifier is None:
                return
            classifier.description = description

    def delete_classifier(self, name):
        with self.session_scope() as session:
            classifier_type = self.get_classifier_type(name)
            if classifier_type == SINGLE_LABEL:
                classifier = session.query(Classifier).filter(Classifier.name == name).scalar()
                if classifier is not None:
                    session.delete(classifier)
            elif classifier_type == MULTI_LABEL:
                labels = self.get_classifier_labels(name)
                for label in labels:
                    classifier = session.query(Classifier).filter(
                        Classifier.name == name + _CLASSIFIER_LABEL_SEP + label).scalar()
                    if classifier is not None:
                        session.delete(classifier)
                classifier = session.query(Classifier).filter(Classifier.name == name).scalar()
                if classifier is not None:
                    session.delete(classifier)

    def update_classifier_model(self, name, model):
        with self.session_scope() as session:
            try:
                classifier = session.query(Classifier).filter(Classifier.name == name).scalar()
                classifier.model = model
                classifier.classifier_type = get_classifier_type_from_model(model)
            except (OperationalError, MemoryError) as e:
                session.rollback()
                raise e

    def create_training_examples(self, classifier_name, text_and_labels):
        with self.session_scope() as session:
            try:
                text_set = list(set((text for text, _ in text_and_labels)))
                stmt = insert(TrainingDocument.__table__).values(
                    [{'text': text,
                      'md5': md5(text.encode('utf-8')).hexdigest()} for text in text_set])
                stmt = stmt.on_conflict_do_update(index_elements=['md5'],
                                                  set_={'creation': datetime.datetime.now()})
                stmt = stmt.returning(TrainingDocument.id)
                with self._lock_training_documents():
                    text_set_ids = [id[0] for id in session.execute(stmt)]

                text_id_map = dict()
                for text, text_set_id in zip(text_set, text_set_ids):
                    text_id_map[text] = text_set_id

                classifier_id = session.query(Classifier.id).filter(Classifier.name == classifier_name).scalar()

                classifier_type = self.get_classifier_type(classifier_name)

                if classifier_type == SINGLE_LABEL:
                    label_set = set((label for _, label in text_and_labels))

                    label_id_map = dict()

                    for label in label_set:
                        label_id = (session.query(Label.id)
                                    .filter(Label.classifier_id == classifier_id)
                                    .filter(Label.name == label)
                                    .scalar()
                                    )
                        label_id_map[label] = label_id

                    text_id_and_label_ids = dict()
                    for text, label in text_and_labels:
                        text_id_and_label_ids[text_id_map[text]] = label_id_map[label]

                    text_id_and_label_ids = text_id_and_label_ids.items()

                    values = [{'document_id': document_id,
                               'classifier_id': classifier_id,
                               'label_id': label_id} for document_id, label_id in text_id_and_label_ids]

                    stmt = insert(Classification.__table__).values(
                        values)
                    stmt = stmt.on_conflict_do_update(index_elements=['document_id', 'classifier_id'],
                                                      set_=dict(stmt.excluded._data))
                    session.execute(stmt)
                elif classifier_type == MULTI_LABEL:
                    label_set = set((label.split(':')[0] for _, label in text_and_labels))
                    label_classifiers_id_map = dict()

                    for label in label_set:
                        label_classifier_id = (session.query(Classifier.id)
                                               .filter(
                            Classifier.name == classifier_name + _CLASSIFIER_LABEL_SEP + label)
                                               .scalar()
                                               )
                        yes_label_id = (session.query(Label.id)
                                        .filter(Label.classifier_id == label_classifier_id)
                                        .filter(Label.name == YES_LABEL)
                                        .scalar()
                                        )
                        no_label_id = (session.query(Label.id)
                                       .filter(Label.classifier_id == label_classifier_id)
                                       .filter(Label.name == NO_LABEL)
                                       .scalar()
                                       )
                        label_classifiers_id_map[label] = label_classifier_id
                        label_classifiers_id_map[label + NO_LABEL] = no_label_id
                        label_classifiers_id_map[label + YES_LABEL] = yes_label_id

                    text_id_and_label_ids = list()
                    for text, label_value in text_and_labels:
                        label, value = label_value.split(':')
                        label_classifier_id = label_classifiers_id_map[label]
                        if value == YES_LABEL:
                            text_id_and_label_ids.append(
                                (text_id_map[text], label_classifier_id, label_classifiers_id_map[label + YES_LABEL]))
                        elif value == NO_LABEL:
                            text_id_and_label_ids.append(
                                (text_id_map[text], label_classifier_id, label_classifiers_id_map[label + NO_LABEL]))

                    values = [{'document_id': document_id,
                               'classifier_id': label_classifier_id,
                               'label_id': label_id} for document_id, label_classifier_id, label_id in
                              text_id_and_label_ids]

                    dummy_label_id = (session.query(Label.id)
                                      .filter(Label.classifier_id == classifier_id)
                                      .filter(Label.name == list(label_set)[0])
                                      .scalar()
                                      )

                    values.extend([{'document_id': document_id,
                                    'classifier_id': classifier_id,
                                    'label_id': dummy_label_id} for document_id in
                                   set([document_id for document_id, _, _ in text_id_and_label_ids])])

                    stmt = insert(Classification.__table__).values(
                        values)
                    stmt = stmt.on_conflict_do_update(index_elements=['document_id', 'classifier_id'],
                                                      set_=dict(stmt.excluded._data))
                    session.execute(stmt)
            except (OperationalError, MemoryError) as e:
                session.rollback()
                raise e

    def classify(self, classifier_name, X, mark_human_labels=False):
        clf = self.get_classifier_model(classifier_name)
        y = list()
        classifier_type = self.get_classifier_type(classifier_name)
        if classifier_type == SINGLE_LABEL:
            if clf is None:
                default_label = self.get_classifier_labels(classifier_name)[0]
            for x in X:
                label = self.get_label(classifier_name, x)
                if label is not None:
                    if mark_human_labels:
                        y.append(_HUMAN_MADE + label)
                    else:
                        y.append(label)
                elif clf is not None:
                    y.append(clf.predict([x])[0])
                else:
                    y.append(default_label)
        elif classifier_type == MULTI_LABEL:
            for x in X:
                y_x = list()
                known_labels_assignment = self.get_label(classifier_name, x)
                if known_labels_assignment is None:
                    known_labels = {}
                else:
                    if mark_human_labels:
                        known_labels = {label_value.split(':')[0]: _HUMAN_MADE + label_value for label_value in
                                        known_labels_assignment}
                    else:
                        known_labels = {label_value.split(':')[0]: label_value for label_value in
                                        known_labels_assignment}
                if clf is None:
                    pred_labels = {}
                else:
                    pred_labels = {label_value.split(':')[0]: label_value for label_value in clf.predict([x])[0]}
                for label in clf.labels():
                    if label in known_labels:
                        y_x.append(known_labels[label])
                    elif label in pred_labels:
                        y_x.append(pred_labels[label])
                    else:
                        y_x.append(label + ':no')
                y.append(y_x)
        return y

    def score(self, classifier_name, X):
        clf = self.get_classifier_model(classifier_name)
        if clf is None:
            labels = self.get_classifier_labels(classifier_name)
            if self.get_classifier_type(classifier_name) == SINGLE_LABEL:
                equi_prob = {label: 1 / len(labels) for label in labels}
            else:
                equi_prob = {label: 1 / 2 for label in labels}
            return [equi_prob for _ in X]
        scores = clf.decision_function(X)
        labels = clf.labels()
        return [dict(zip(labels, values)) for values in scores]

    def get_label(self, classifier_name, text):
        classifier_type = self.get_classifier_type(classifier_name)
        if classifier_type == SINGLE_LABEL:
            with self.session_scope() as session:
                return (session.query(Label.name)
                        .filter(TrainingDocument.md5 == md5(text.encode('utf-8')).hexdigest())
                        .join(Classification.document)
                        .join(Classification.classifier)
                        .join(Classification.label)
                        .filter(Classifier.name == classifier_name)
                        .scalar()
                        )
        elif classifier_type == MULTI_LABEL:
            labels = self.get_classifier_labels(classifier_name)
            assigned_labels = list()
            with self.session_scope() as session:
                for label in labels:
                    assigned = (session.query(Label.name)
                                .filter(TrainingDocument.md5 == md5(text.encode('utf-8')).hexdigest())
                                .join(Classification.document)
                                .join(Classification.classifier)
                                .join(Classification.label)
                                .filter(Classifier.name == classifier_name + _CLASSIFIER_LABEL_SEP + label)
                                .scalar()
                                )
                    if assigned:
                        assigned_labels.append(label + ':' + assigned)
            return assigned_labels

    def rename_classifier_label(self, classifier_name, label_name, new_name):
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
        clf = self.get_classifier_model(classifier_name)
        if clf is not None:
            clf.rename_label(label_name, new_name)
            self.update_classifier_model(classifier_name, clf)
        classifier_type = self.get_classifier_type(classifier_name)
        if classifier_type == MULTI_LABEL:
            classifier = session.query(Classifier).filter(
                Classifier.name == classifier_name + _CLASSIFIER_LABEL_SEP + label_name).scalar()
            classifier.name = classifier_name + _CLASSIFIER_LABEL_SEP + new_name

    def dataset_names(self):
        with self.session_scope() as session:
            return self._flatten_list(session.query(Dataset.name).order_by(Dataset.name))

    def dataset_exists(self, name):
        with self.session_scope() as session:
            return session.query(exists().where(Dataset.name == name)).scalar()

    def create_dataset(self, name):
        with self.session_scope() as session:
            dataset = Dataset(name)
            session.add(dataset)

    def rename_dataset(self, name, newname):
        with self.session_scope() as session:
            dataset = session.query(Dataset).filter(Dataset.name == name).scalar()
            if dataset is None:
                return
            dataset.name = newname

    def delete_dataset(self, name):
        with self.session_scope() as session:
            dataset = session.query(Dataset).filter(Dataset.name == name).scalar()
            if dataset is not None:
                session.delete(dataset)

    def get_dataset_creation_time(self, name):
        with self.session_scope() as session:
            return session.query(Dataset.creation).filter(Dataset.name == name).scalar()

    def get_dataset_last_update_time(self, name):
        with self.session_scope() as session:
            return session.query(Dataset.last_updated).filter(Dataset.name == name).scalar()

    def get_dataset_size(self, name):
        with self.session_scope() as session:
            return session.query(Dataset.documents).filter(Dataset.name == name).filter(
                DatasetDocument.dataset_id == Dataset.id).count()

    def create_dataset_documents(self, dataset_name, external_ids_and_contents):
        with self.session_scope() as session:
            dataset = session.query(Dataset).filter(Dataset.name == dataset_name).one()
            values = dict()
            for external_id, content in external_ids_and_contents:
                values[external_id] = {'text': content,
                                       'md5': md5(content.encode('utf-8')).hexdigest(),
                                       'dataset_id': dataset.id,
                                       'external_id': external_id}
            stmt = insert(DatasetDocument.__table__).values(list(values.values()))
            stmt = stmt.on_conflict_do_update(index_elements=['dataset_id', 'external_id'],
                                              set_=dict(stmt.excluded._data))
            ret = session.execute(stmt)
            dataset.last_updated = datetime.datetime.now()

    def delete_dataset_document(self, dataset_name, external_id):
        with self.session_scope() as session:
            dataset = session.query(Dataset).filter(Dataset.name == dataset_name).one()
            document = (session.query(DatasetDocument)
                        .filter(DatasetDocument.dataset_id == dataset.id)
                        .filter(DatasetDocument.external_id == external_id)
                        .scalar()
                        )
            if document is not None:
                session.delete(document)
            dataset.last_updated = datetime.datetime.now()

    def get_classifier_examples_count(self, name):
        with self.session_scope() as session:
            count = (session.query(Classification.id)
                     .filter(Classifier.name == name)
                     .join(Classification.classifier)
                     .count()
                     )
            if count:
                return count
            else:
                return 0

    def get_classifier_examples_with_label_count(self, name, label):
        classifier_type = self.get_classifier_type(name)
        if classifier_type == SINGLE_LABEL or classifier_type == _CLASSIFIER_TYPE_HIDDEN_FOR_MULTI_LABEL:
            with self.session_scope() as session:
                count = (session.query(Classification.id)
                         .filter(Classifier.name == name)
                         .join(Classification.classifier)
                         .join(Classification.label)
                         .filter(Label.name == label)
                         .count())
            if count:
                return count
            else:
                return 0
        elif classifier_type == MULTI_LABEL:
            with self.session_scope() as session:
                count = (session.query(Classification.id)
                         .filter(Classifier.name == name + _CLASSIFIER_LABEL_SEP + label)
                         .join(Classification.classifier)
                         .join(Classification.label)
                         .filter(Label.name == label)
                         .count())
            if count:
                return count
            else:
                return 0

    def get_classifier_examples(self, name, offset=0, limit=None):
        with self.session_scope() as session:
            return (session.query(Classification)
                    .filter(Classifier.name == name)
                    .join(Classification.classifier)
                    .order_by(Classification.id)
                    .offset(offset)
                    .limit(limit)
                    )

    def get_classifier_examples_with_label(self, name, label, offset=0, limit=None):
        classifier_type = self.get_classifier_type(name)
        if classifier_type == SINGLE_LABEL or classifier_type == _CLASSIFIER_TYPE_HIDDEN_FOR_MULTI_LABEL:
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
        elif classifier_type == MULTI_LABEL:
            with self.session_scope() as session:
                return (session.query(Classification)
                        .filter(Classifier.name == name)
                        .join(Classification.classifier)
                        .join(Classification.label)
                        .filter(Label.name == name + _CLASSIFIER_LABEL_SEP + label)
                        .order_by(Classification.id)
                        .offset(offset)
                        .limit(limit)
                        )

    def get_dataset_random_documents_without_labels(self, dataset_name, classifier_name, filter, limit=None,
                                                    require_all_labels_for_multi_label=True):
        classifier_type = self.get_classifier_type(classifier_name)
        if classifier_type == SINGLE_LABEL or not require_all_labels_for_multi_label:
            with self.session_scope() as session:
                return (session.query(DatasetDocument.text,DatasetDocument.id)
                        .join(DatasetDocument.dataset)
                        .filter(Dataset.name == dataset_name)
                        .filter(DatasetDocument.text.like('%' + filter + '%'))
                        .filter(not_(exists().where(and_(DatasetDocument.md5 == TrainingDocument.md5,
                                                         Classification.classifier_id == Classifier.id,
                                                         Classifier.name == classifier_name,
                                                         TrainingDocument.id == Classification.document_id))))
                        .order_by(func.random())
                        .limit(limit)
                        )
        elif classifier_type == MULTI_LABEL:
            labels = self.get_classifier_labels(classifier_name)
            label_classifiers = [classifier_name + _CLASSIFIER_LABEL_SEP + label for label in labels]
            with self.session_scope() as session:
                missing = list(session.query(DatasetDocument.text,DatasetDocument.id)
                               .join(DatasetDocument.dataset)
                               .filter(Dataset.name == dataset_name)
                               .filter(DatasetDocument.text.like('%' + filter + '%'))
                               .filter(not_(exists().where(and_(DatasetDocument.md5 == TrainingDocument.md5,
                                                                Classification.classifier_id == Classifier.id,
                                                                Classifier.name == classifier_name,
                                                                TrainingDocument.id == Classification.document_id))))
                               .order_by(func.random())
                               .limit(limit)
                               )
                if len(missing) > 0:
                    return missing

                partial = list(session.query(DatasetDocument.text,DatasetDocument.id)
                               .join(DatasetDocument.dataset)
                               .filter(Dataset.name == dataset_name)
                               .filter(DatasetDocument.text.like('%' + filter + '%'))
                               .filter(DatasetDocument.md5 == TrainingDocument.md5)
                               .filter(TrainingDocument.id == Classification.document_id)
                               .filter(Classification.classifier_id == Classifier.id)
                               .filter(Classifier.name.in_(label_classifiers))
                               .group_by(DatasetDocument.id)
                               .having(count(Classification.id) != len(label_classifiers))
                               .order_by(func.random())
                               .limit(limit)
                               )
                return partial

    def get_dataset_random_documents(self, dataset_name, filter, limit=None):
        with self.session_scope() as session:
            return (session.query(DatasetDocument)
                    .join(DatasetDocument.dataset)
                    .filter(Dataset.name == dataset_name)
                    .filter(DatasetDocument.text.like('%' + filter + '%'))
                    .order_by(func.random())
                    .limit(limit)
                    )

    def get_dataset_documents_without_labels_count(self, dataset_name, classifier_name,
                                                   require_all_labels_for_multi_label=True):
        classifier_type = self.get_classifier_type(classifier_name)
        if classifier_type == SINGLE_LABEL or not require_all_labels_for_multi_label:
            with self.session_scope() as session:
                return (session.query(DatasetDocument.id)
                        .join(DatasetDocument.dataset)
                        .filter(Dataset.name == dataset_name)
                        .filter(not_(exists().where(and_(DatasetDocument.md5 == TrainingDocument.md5,
                                                         Classification.classifier_id == Classifier.id,
                                                         Classifier.name == classifier_name,
                                                         TrainingDocument.id == Classification.document_id))))
                        .count()
                        )

        elif classifier_type == MULTI_LABEL:
            labels = self.get_classifier_labels(classifier_name)
            label_classifiers = [classifier_name + _CLASSIFIER_LABEL_SEP + label for label in labels]
            with self.session_scope() as session:
                count_partial = (session.query(DatasetDocument.id)
                                 .join(DatasetDocument.dataset)
                                 .filter(Dataset.name == dataset_name)
                                 .filter(DatasetDocument.md5 == TrainingDocument.md5)
                                 .filter(TrainingDocument.id == Classification.document_id)
                                 .filter(Classification.classifier_id == Classifier.id)
                                 .filter(Classifier.name.in_(label_classifiers))
                                 .group_by(DatasetDocument.id)
                                 .having(count(Classification.id) != len(label_classifiers))
                                 .count()
                                 )
                count_missing = (session.query(DatasetDocument.id)
                                 .join(DatasetDocument.dataset)
                                 .filter(Dataset.name == dataset_name)
                                 .filter(not_(exists().where(and_(DatasetDocument.md5 == TrainingDocument.md5,
                                                                  Classification.classifier_id == Classifier.id,
                                                                  Classifier.name == classifier_name,
                                                                  TrainingDocument.id == Classification.document_id))))
                                 .count()
                                 )
                return count_partial + count_missing

    def get_dataset_documents_sorted_by_name(self, name, offset=0, limit=None):
        with self.session_scope() as session:
            return (session.query(DatasetDocument)
                    .filter(Dataset.name == name)
                    .join(DatasetDocument.dataset)
                    .order_by(DatasetDocument.external_id)
                    .offset(offset)
                    .limit(limit)
                    )

    def get_dataset_documents_sorted_by_position(self, name, offset=0, limit=None):
        with self.session_scope() as session:
            return (session.query(DatasetDocument)
                    .filter(Dataset.name == name)
                    .join(DatasetDocument.dataset)
                    .order_by(DatasetDocument.id)
                    .offset(offset)
                    .limit(limit)
                    )

    def get_dataset_document_by_name(self, datasetname, documentname):
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

    def get_dataset_document_by_position(self, name, position):
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

    def get_dataset_document_position_by_id(self, name, document_id):
        with self.session_scope() as session:
            return (session.query(DatasetDocument)
                    .filter(Dataset.name == name)
                    .join(DatasetDocument.dataset)
                    .filter(DatasetDocument.id < document_id)
                    .order_by(DatasetDocument.id)
                    .count()
                    )

    def get_jobs(self, starttime=None):
        if starttime is None:
            starttime = datetime.datetime.now() - datetime.timedelta(days=1)
        with self.session_scope() as session:
            return (session.query(Job)
                    .filter(Job.creation > starttime)
                    .order_by(Job.creation.desc())
                    )

    def get_locks(self):
        with self.session_scope() as session:
            return session.query(Lock).order_by(Lock.creation.desc())

    def create_job(self, function, args=(), kwargs=None, description=None):
        if kwargs is None:
            kwargs = {}
        with self.session_scope() as session:
            if description is None:
                description = function.__name__
            job = Job(description, {'function': function, 'args': args, 'kwargs': kwargs})
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

    def set_job_completion_time(self, job_id, completion=None):
        if completion is None:
            completion = datetime.datetime.now()
        with self.session_scope() as session:
            job = session.query(Job).filter(Job.id == job_id).one()
            job.completion = completion

    def set_job_start_time(self, job_id, start=None):
        if start is None:
            start = datetime.datetime.now()
        with self.session_scope() as session:
            job = session.query(Job).filter(Job.id == job_id).one()
            job.start = start

    def set_job_status(self, job_id, status):
        with self.session_scope() as session:
            job = session.query(Job).filter(Job.id == job_id).one()
            job.status = status

    def get_job_status(self, job_id):
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

    def create_classification_job(self, datasetname, classifiers, job_id, fullpath):
        with self.session_scope() as session:
            dataset_id = session.query(Dataset.id).filter(Dataset.name == datasetname).scalar()
            classification_job = ClassificationJob(dataset_id, ', '.join(classifiers), job_id, fullpath)
            session.add(classification_job)

    def get_classification_jobs(self, name):
        with self.session_scope() as session:
            return (session.query(ClassificationJob)
                    .join(ClassificationJob.dataset)
                    .filter(Dataset.name == name)
                    .order_by(ClassificationJob.creation)
                    )

    def get_classification_job_filename(self, id):
        with self.session_scope() as session:
            return (session.query(ClassificationJob.filename)
                    .filter(ClassificationJob.id == id)
                    .scalar()
                    )

    def delete_classification_job(self, id):
        with self.session_scope() as session:
            classification_job = session.query(ClassificationJob).filter(
                ClassificationJob.id == id).scalar()
            if os.path.exists(classification_job.filename):
                os.remove(classification_job.filename)
            if classification_job is not None:
                session.delete(classification_job)

    def delete_job(self, id):
        with self.session_scope() as session:
            job = session.query(Job).filter(Job.id == id).scalar()
            if job is not None:
                session.delete(job)

    def delete_lock(self, name):
        with self.session_scope() as session:
            lock = session.query(Lock).filter(Lock.name == name).scalar()
            if lock is not None:
                session.delete(lock)

    def classification_exists(self, filename):
        with self.session_scope() as session:
            return session.query(exists().where(ClassificationJob.filename == filename)).scalar()

    def job_exists(self, id):
        with self.session_scope() as session:
            return session.query(exists().where(Job.id == id)).scalar()

    def ipaddresses(self):
        with self.session_scope() as session:
            return self._flatten_list(session.query(IPTracker.ip).order_by(IPTracker.ip))

    def get_iptracker_creation_time(self, ip):
        with self.session_scope() as session:
            return session.query(IPTracker.creation).filter(IPTracker.ip == ip).scalar()

    def get_iptracker_last_update_time(self, ip):
        with self.session_scope() as session:
            return session.query(IPTracker.last_updated).filter(IPTracker.ip == ip).scalar()

    def get_iptracker_hourly_limit(self, ip):
        with self.session_scope() as session:
            return session.query(IPTracker.hourly_limit).filter(IPTracker.ip == ip).scalar()

    def set_iptracker_hourly_limit(self, ip, hourly_limit):
        with self.session_scope() as session:
            iptracker = session.query(IPTracker).filter(IPTracker.ip == ip).scalar()
            iptracker.hourly_limit = hourly_limit

    def get_iptracker_request_limit(self, ip):
        with self.session_scope() as session:
            return session.query(IPTracker.request_limit).filter(IPTracker.ip == ip).scalar()

    def set_iptracker_request_limit(self, ip, request_limit):
        with self.session_scope() as session:
            iptracker = session.query(IPTracker).filter(IPTracker.ip == ip).scalar()
            iptracker.request_limit = request_limit

    def get_iptracker_total_request_counter(self, ip):
        with self.session_scope() as session:
            return session.query(IPTracker.total_request_counter).filter(IPTracker.ip == ip).scalar()

    def get_iptracker_current_request_counter(self, ip):
        with self.session_scope() as session:
            iptracker = session.query(IPTracker).filter(IPTracker.ip == ip).scalar()
            return iptracker.check_current_request_counter()

    def set_iptracker_current_request_counter(self, ip, count):
        with self.session_scope() as session:
            iptracker = session.query(IPTracker).filter(IPTracker.ip == ip).scalar()
            iptracker.current_request_counter = count

    def iptracker_check_and_count_request(self, ip, cost):
        with self.session_scope() as session:
            iptracker = session.query(IPTracker).filter(IPTracker.ip == ip).scalar()
            if iptracker is None:
                raise LookupError()
            return iptracker.check_and_count_request(cost)

    def create_iptracker(self, ip, hourly_limit, request_limit):
        with self.session_scope() as session:
            iptracker = IPTracker(ip, hourly_limit, request_limit)
            session.add(iptracker)

    def delete_iptracker(self, ip):
        with self.session_scope() as session:
            ip = session.query(IPTracker).filter(IPTracker.ip == ip).scalar()
            if ip is not None:
                session.delete(ip)

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

    def keys(self):
        with self.session_scope() as session:
            return self._flatten_list(session.query(KeyTracker.key).order_by(KeyTracker.key))

    def get_keytracker_name(self, key):
        with self.session_scope() as session:
            return session.query(KeyTracker.name).filter(KeyTracker.key == key).scalar()

    def get_keytracker_creation_time(self, key):
        with self.session_scope() as session:
            return session.query(KeyTracker.creation).filter(KeyTracker.key == key).scalar()

    def get_keytracker_last_update_time(self, key):
        with self.session_scope() as session:
            return session.query(KeyTracker.last_updated).filter(KeyTracker.key == key).scalar()

    def get_keytracker_hourly_limit(self, key):
        with self.session_scope() as session:
            return session.query(KeyTracker.hourly_limit).filter(KeyTracker.key == key).scalar()

    def get_keytracker_request_limit(self, key):
        with self.session_scope() as session:
            return session.query(KeyTracker.request_limit).filter(KeyTracker.key == key).scalar()

    def get_keytracker_current_request_counter(self, key):
        with self.session_scope() as session:
            keytracker = session.query(KeyTracker).filter(KeyTracker.key == key).scalar()
            return keytracker.check_current_request_counter()

    def get_keytracker_total_request_counter(self, key):
        with self.session_scope() as session:
            return session.query(KeyTracker.total_request_counter).filter(KeyTracker.key == key).scalar()

    def keytracker_check_and_count_request(self, key, cost):
        with self.session_scope() as session:
            key_obj = session.query(KeyTracker).filter(KeyTracker.key == key).scalar()
            if key_obj is None:
                return False
            return key_obj.check_and_count_request(cost)

    def set_keytracker_hourly_limit(self, key, hourly_limit):
        with self.session_scope() as session:
            key_obj = session.query(KeyTracker).filter(KeyTracker.key == key).scalar()
            key_obj.hourly_limit = hourly_limit

    def set_keytracker_request_limit(self, key, request_limit):
        with self.session_scope() as session:
            key_obj = session.query(KeyTracker).filter(KeyTracker.key == key).scalar()
            key_obj.request_limit = request_limit

    def set_keytracker_current_request_counter(self, key, count):
        with self.session_scope() as session:
            key_obj = session.query(KeyTracker).filter(KeyTracker.key == key).scalar()
            key_obj.current_request_counter = count

    def create_keytracker(self, name, hourly_limit, request_limit):
        with self.session_scope() as session:
            key_obj = KeyTracker(name, hourly_limit, request_limit)
            session.add(key_obj)
            return key_obj.key

    def delete_keytracker(self, key):
        with self.session_scope() as session:
            key = session.query(KeyTracker).filter(KeyTracker.key == key).scalar()
            if key is not None:
                session.delete(key)

    def _flatten_list(self, list_of_list):
        return [item for sublist in list_of_list for item in sublist]

    def _lock_training_documents(self):
        return DBLock(self, 'training documents')

    @staticmethod
    def version():
        return "4.1.1"


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
