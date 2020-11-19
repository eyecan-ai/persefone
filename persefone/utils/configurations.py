from box.from_file import converters
from box import box_from_file, Box, BoxList
import pydash
from typing import Any, Dict, Sequence, Tuple
from schema import Schema
from pathlib import Path


class XConfiguration(Box):

    def __init__(self, filename=None):
        if filename is not None:
            self.__params = box_from_file(file=Path(filename))
        else:
            self.__params = Box()
        self.__schema = None

    @property
    def params(self):
        return self.__params

    def set_schema(self, schema: Schema):
        assert isinstance(schema, Schema), "schema is not a valid Schema object!"
        self.__schema = schema

    def get_schema(self) -> Schema:
        return self.__schema

    def validate(self):
        schema = self.get_schema()
        if schema is not None:
            self.get_schema().validate(self.params.to_dict())

    def is_valid(self):
        schema = self.get_schema()
        if schema is not None:
            return schema.is_valid(self.params.to_dict())
        return True

    def save_to(self, filename):
        filename = Path(filename)
        if 'yml' or 'yaml' in filename.suffix.lower():
            self.params.to_yaml(filename=filename)
        elif 'json' in filename.suffix.lower():
            self.params.to_json(filename=filename)
        else:
            raise NotImplementedError(f"Extension {filename.suffix.lower()} not supported yet!")

    @classmethod
    def from_dict(cls, d):
        cfg = XConfiguration()
        cfg.__params = Box(d)
        return cfg


class YConfiguration(Box):
    PRIVATE_QUALIFIER = '_'
    REFERENCE_QUALIFIER = '@'
    REPLACE_QUALIFIER = '$'
    KNOWN_EXTENSIONS = converters.keys()

    def __init__(self, filename: str = None):
        """ Creates a YConfiguration object from configuration file

        :param filename: configuration file [yaml, json, toml], defaults to None
        :type filename: str, optional
        """
        self._filename = None
        if filename is not None:
            self._filename = Path(filename)
            self.update(box_from_file(file=Path(filename)))
        self._schema = None
        self.deep_parse()

    def get_schema(self):
        return self._schema

    def set_schema(self, s: Schema):
        """ Push validation schema

        :param schema: validation schema
        :type schema: Schema
        """
        if s is not None:
            assert isinstance(s, Schema), "schema is not a valid Schema object!"
        self._schema: Schema = s

    def validate(self):
        """ Validate internal schema if any """
        if self.get_schema() is not None:
            self.get_schema().validate(self.to_dict())

    def is_valid(self) -> bool:
        """ Check for schema validity

        :return: TRUE for valid or no schema inside
        :rtype: bool
        """
        if self.get_schema() is not None:
            return self.get_schema().is_valid(self.to_dict())
        return True

    def save_to(self, filename: str):
        """ Save configuration to output file

        :param filename: output filename
        :type filename: str
        :raises NotImplementedError: Raise error for unrecognized extension
        """
        filename = Path(filename)
        if 'yml' in filename.suffix.lower() or 'yaml' in filename.suffix.lower():
            self.to_yaml(filename=filename)
        elif 'json' in filename.suffix.lower():
            self.to_json(filename=filename)
        elif 'toml' in filename.suffix.lower():
            self.to_toml(filename=filename)
        else:
            raise NotImplementedError(f"Extension {filename.suffix.lower()} not supported yet!")

    def chunks(self, discard_private_qualifiers: bool = True) -> Sequence[Tuple[str, Any]]:
        """ Builds a plain view of dictionary with pydash notation

        :param discard_private_qualifiers: TRUE to discard keys starting with private qualifier, defaults to True
        :type discard_private_qualifiers: bool, optional
        :return: list of pairs (key, value) where key is a dot notation pydash key (e.g. d['one']['two']['three'] -> 'one.two.three' )
        :rtype: Sequence[Tuple[str, Any]]
        """
        return self._walk(self, discard_private_qualifiers=discard_private_qualifiers)

    def replace(self, old_value: str, new_value: str):
        """ Replaces target value with custom new value

        :param old_value: value to replace
        :type old_value: str
        :param new_value: new key value
        :type new_value: str
        """
        if old_value.startswith(self.REPLACE_QUALIFIER):
            chunks = self.chunks(discard_private_qualifiers=True)
            for k, v in chunks:
                if v == old_value:
                    pydash.set_(self, k, new_value)

    def replace_map(self, m: dict):
        """ Replace target old values with new values represented as dict

        :param m: dict of key/value = old/new
        :type m: dict
        """

        for old_v, new_v in m.items():
            self.replace(old_v, new_v)

    def deep_parse(self):
        """ Deep visit of dictionary replacing filename values with a new YConfiguration object recusively
        """
        chunks = self.chunks()
        for chunk_name, value in chunks:
            if self._could_be_path(value):
                p = Path(value.replace(self.REFERENCE_QUALIFIER, ''))
                if self._filename is not None and not p.is_absolute():
                    p = self._filename.parent / p

                if p.exists():
                    sub_cfg = YConfiguration(filename=p)
                    pydash.set_(self, chunk_name, sub_cfg)
                else:
                    raise OSError(f'File {p} not found!')

    def _could_be_path(self, p: str) -> bool:
        """ Check if a string could be a path. It's not a robust test outside YConfiguration!

        :param p: source string
        :type p: str
        :return: TRUE = 'maybe is path'
        :rtype: bool
        """
        if isinstance(p, str):
            if any(f'.{x}{self.REFERENCE_QUALIFIER}' in p for x in self.KNOWN_EXTENSIONS):
                return True
        return False

    def to_dict(self, discard_private_qualifiers: bool = True) -> Dict:
        """
        Turn the Box and sub Boxes back into a native python dictionary.

        :return: python dictionary of this Box
        """
        out_dict = dict(self)
        for k, v in out_dict.items():
            if v is self:
                out_dict[k] = out_dict
            elif isinstance(v, Box):
                out_dict[k] = v.to_dict()
            elif isinstance(v, BoxList):
                out_dict[k] = v.to_list()

        if discard_private_qualifiers:
            chunks = self.chunks(discard_private_qualifiers=False)
            for chunk_name, value in chunks:
                if f'.{self.PRIVATE_QUALIFIER}' in chunk_name or chunk_name.startswith(self.PRIVATE_QUALIFIER):
                    pydash.unset(out_dict, chunk_name)
        return out_dict

    @classmethod
    def from_dict(cls, d: dict) -> 'YConfiguration':
        """ Creates YConfiguration from a plain dictionary

        :param d: input dictionary
        :type d: dict
        :return: built YConfiguration
        :rtype: YConfiguration
        """
        cfg = YConfiguration()
        cfg.update(Box(d))
        return cfg

    @classmethod
    def _walk(cls,
              d: Dict, path: Sequence = None,
              chunks: Sequence = None,
              discard_private_qualifiers: bool = True) -> Sequence[Tuple[str, Any]]:
        """ Deep visit of dictionary building a plain sequence of pairs (key,value) where key has a pydash notation

        :param d: input dictionary
        :type d: Dict
        :param path: private output value for path (not use), defaults to None
        :type path: Sequence, optional
        :param chunks: private output to be fileld with retrieved pairs (not use), defaults to None
        :type chunks: Sequence, optional
        :param discard_private_qualifiers: TRUE to discard keys starting with private qualifier, defaults to True
        :type discard_private_qualifiers: bool, optional
        :return: sequence of retrieved pairs
        :rtype: Sequence[Tuple[str, Any]]
        """
        root = False
        if path is None:
            path, chunks, root = [], [], True
        for k, v in d.items():
            if isinstance(v, dict):
                path.append(k)
                cls._walk(v, path=path, chunks=chunks, discard_private_qualifiers=discard_private_qualifiers)
                path.pop()
            else:
                path.append(k)
                chunk_name = ".".join(path)
                if not(discard_private_qualifiers and chunk_name.startswith(cls.PRIVATE_QUALIFIER)):
                    chunks.append((chunk_name, v))
                path.pop()
        if root:
            return chunks
