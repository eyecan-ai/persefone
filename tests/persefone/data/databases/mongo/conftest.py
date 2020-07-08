import pytest
from mongoengine import connect, disconnect
import logging


@pytest.fixture(scope='function')
def temp_mongo_database_keep_alive():
    db_name = '_perse_'
    try:
        db = connect(db_name, serverSelectionTimeoutMS=1000)
        yield db
        disconnect()
    except Exception as e:
        logging.error(e)
        disconnect()
        pass


@pytest.fixture(scope='function')
def temp_mongo_database():
    db_name = '##_temp_database_@@'
    try:
        db = connect(db_name, serverSelectionTimeoutMS=1000)
        yield db
        db.drop_database(db_name)
        disconnect()
    except Exception as e:
        logging.error(e)
        disconnect()
        pass


@pytest.fixture(scope='function')
def temp_mongo_mock_database():
    db_name = '##_temp_mock_database_@@'
    db = connect(db_name, host='mongomock://localhost')
    yield db
    db.drop_database(db_name)
    disconnect()
