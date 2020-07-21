from schema import Schema
from box import Box, box_from_file
from pathlib import Path

# TODO: add documentation


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
