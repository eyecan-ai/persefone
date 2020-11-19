import copy
from box.box_list import BoxList
import numpy as np
from typing import Union
from box.box import Box
from deepdiff import DeepDiff
from schema import Schema, Or, Optional, Regex, SchemaMissingKeyError
import uuid

import pydash
from persefone.utils.configurations import XConfiguration, YConfiguration
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


class TestYConfiguration(object):

    def _store_cfg(self, filename: Union[str, Path], d: dict):

        filename = str(filename)
        data = Box(d)
        if 'yml' in filename or 'yaml' in filename:
            data.to_yaml(filename)
        if 'json' in filename:
            data.to_json(filename)
        if 'toml' in filename:
            data.to_toml(filename)

    def _sample_dict(self):
        np.random.seed(666)

        placeholders = [
            f'{YConfiguration.REPLACE_QUALIFIER}v_0',
            f'{YConfiguration.REPLACE_QUALIFIER}v_1',
            f'{YConfiguration.REPLACE_QUALIFIER}v_2',
            f'{YConfiguration.REPLACE_QUALIFIER}v_3'
        ]

        sample_dict = {
            'one': placeholders[0],
            'two': np.random.randint(-50, 50, (32, 32)).tolist(),
            'three': {
                '3.1': True,
                '2.1': [False, False],
                'john': {
                    'john1': placeholders[1],
                    'john2': 2,
                    'john3': {
                        'pino': [3.3, 3.3],
                        'mino': {
                            'a': [True, False]
                        }
                    }
                }
            },
            'first': {
                'f1': placeholders[2],
                'f2': 2.22,
                'f3': [3, 3, 3, 3, 3, 3],
                'external': {
                    'ext': np.random.uniform(-2, 2, (7, 12)).tolist(),
                    'ext_name': placeholders[3],
                }
            }
        }

        schema = Schema(
            {
                'one': Or(int, str),
                'two': list,
                'three': {
                    '3.1': bool,
                    '2.1': [bool],
                    'john': dict
                },
                'first': {
                    Regex(''): Or(str, int, list, float, dict)
                }
            }
        )

        return sample_dict, schema, placeholders

    def test_creation(self, generic_temp_folder):

        generic_temp_folder = Path(generic_temp_folder)

        extensions = ['yaml', 'json', 'toml']

        for cfg_extension in extensions:

            sample_dict, _, _ = self._sample_dict()
            volatile_dict = copy.deepcopy(sample_dict)
            to_be_raplaced_keys = sorted([
                'three.john',
                'three.john.john3',
                'three.john.john3.mino',
                'first',
                'first.external'
            ], reverse=True)

            subtitutions_values = {}
            for k in to_be_raplaced_keys:
                random_name = str(uuid.uuid1()) + f".{cfg_extension}@"
                # print(k, random_name)
                subtitutions_values[random_name] = pydash.get(sample_dict, k)
                pydash.set_(volatile_dict, k, random_name)

            output_cfg_filename = generic_temp_folder / f'out_config.{cfg_extension}'
            output_cfg_filename2 = generic_temp_folder / f'out_config2.{cfg_extension}'

            subtitutions_values[str(output_cfg_filename)] = volatile_dict
            for output_filename, d in subtitutions_values.items():
                output_filename = generic_temp_folder / output_filename.replace('@', '')
                self._store_cfg(output_filename, d)

            yconf = YConfiguration(output_cfg_filename)
            yconf.save_to(output_cfg_filename2)
            yconf_reloaded = YConfiguration(output_cfg_filename2)

            assert not DeepDiff(yconf.to_dict(), sample_dict)
            assert not DeepDiff(yconf_reloaded.to_dict(), sample_dict)
            assert not DeepDiff(YConfiguration.from_dict(yconf_reloaded.to_dict()).to_dict(), yconf_reloaded.to_dict())
            assert len(YConfiguration.from_dict(yconf_reloaded.to_dict())) > len(sample_dict)  # YConf contains 2 more private keys!

    def test_replace(self, generic_temp_folder):

        sample_dict, _, placeholders = self._sample_dict()

        conf = YConfiguration.from_dict(sample_dict)

        np.random.seed(66)
        to_replace = {}
        for p in placeholders:
            to_replace[p] = np.random.randint(0, 10)

        conf.replace_map(to_replace)

        chunks = conf.chunks()
        for key, value in chunks:
            if not isinstance(value, Box) and not isinstance(value, BoxList):
                assert value not in to_replace.keys()

    def test_validation(self, generic_temp_folder):

        sample_dict, schema, _ = self._sample_dict()

        conf = YConfiguration.from_dict(sample_dict)
        conf.set_schema(schema)

        print(type(conf.get_schema()))
        conf.validate()
        assert conf.is_valid()

        conf.set_schema(None)
        assert conf.is_valid()

        invalid_schema = Schema({'pino': int})
        conf.set_schema(invalid_schema)
        with pytest.raises(SchemaMissingKeyError):
            conf.validate()
        assert not conf.is_valid()
