from persefone.interfaces.proto.comm_pb2 import ResponseStatus


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
