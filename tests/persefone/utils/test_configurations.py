from persefone.utils.configurations import XConfiguration
import schema
from pathlib import Path
import pytest


class TestXConfiguration(object):

    @pytest.fixture()
    def valid_schema(self):
        return schema.Schema({
            'name': str,
            'number': int,
            'listofstrings': schema.And(list, [str]),
            'dictofstrings': schema.And(dict, {str: str}),
            'dictofnumbers': {str: schema.Or(int, float)},
            'nestedobjects': {
                'level1': object,
                'level2': object
            }
        })

    @pytest.fixture()
    def invalid_schema(self):
        return schema.Schema({
            'invalid': str,
            'number': int,
        })

    @pytest.fixture()
    def sample_validable_configuration(self, configurations_folder):
        return Path(configurations_folder) / 'sample_validable_configuration.yml'

    def test_without_schema(self, sample_validable_configuration):

        sample = XConfiguration(filename=sample_validable_configuration)
        sample.validate()
        assert sample.is_valid(), "Should be a valid configuration in no schema is provided!"

        valid_keys = ['name', 'number', 'listofstrings', 'dictofstrings', 'dictofnumbers', 'nestedobjects']

        for key in valid_keys:
            assert key in sample.params, f"key '{key}' is missing in configuration!"

        valid_keys = ['XXX', 'YYY', 'ZZZ']
        for key in valid_keys:
            assert key not in sample.params, f"key '{key}' should be missing!"
            with pytest.raises(KeyError):
                print(sample.params[key])

    def test_with_valid_schema(self, sample_validable_configuration, valid_schema):

        sample = XConfiguration(filename=sample_validable_configuration)
        sample.set_schema(valid_schema)
        sample.validate()
        assert sample.is_valid(), "Should be a valid configuration!"

    def test_with_invalid_schema(self, sample_validable_configuration, invalid_schema):

        sample = XConfiguration(filename=sample_validable_configuration)
        sample.set_schema(invalid_schema)
        with pytest.raises(schema.SchemaMissingKeyError):
            sample.validate()
        assert not sample.is_valid(), "Should be an invalid configuration!"
