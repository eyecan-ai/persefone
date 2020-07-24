from google.protobuf import json_format
from google.protobuf.message import Message
import numpy as np
from persefone.interfaces.proto.comm_pb2 import ResponseStatus
import json


class ResponseStatusUtils(object):
    STATUS_CODE_OK = 0
    STATUS_CODE_ERROR = 1

    @classmethod
    def create_ok_status(cls, message: str = '') -> ResponseStatus:
        return cls.create_status(code=cls.STATUS_CODE_OK, message=message)

    @classmethod
    def create_error_status(cls, message: str = '') -> ResponseStatus:
        return cls.create_status(code=cls.STATUS_CODE_ERROR, message=message)

    @classmethod
    def create_status(cls, code: int, message: str) -> ResponseStatus:
        status = ResponseStatus()
        status.code = code
        status.message = message
        return status


class MetadataUtils(object):

    class NumpyEncoder(json.JSONEncoder):

        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    @classmethod
    def dict_to_struct(cls, metadata: dict, message: Message):
        json_format.Parse(json.dumps(metadata, cls=cls.NumpyEncoder), message)

    @ classmethod
    def struct_to_dict(cls, message: Message):
        return json_format.MessageToDict(message)
