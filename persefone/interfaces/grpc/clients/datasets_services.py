from persefone.interfaces.grpc.datasets_services_pb2_grpc import DatasetsServiceStub
from persefone.interfaces.grpc.datasets_services_pb2 import (
    DDatasetRequest, DDatasetResponse,
    DSampleRequest, DSampleResponse,
    DItemRequest, DItemResponse
)
from google.protobuf import json_format
import grpc
from typing import List, Union, Tuple


class DatasetsServiceClientCFG(object):
    DEFAULT_MAX_MESSAGE_LENGTH = -1

    def __init__(self):
        self.options = [
            ('grpc.max_send_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
        ]


class DatasetsServiceClient(object):

    def __init__(self, host='localhost', port=50051, cfg=DatasetsServiceClientCFG()):
        if isinstance(port, str):
            port = int(port)
        self._channel = grpc.insecure_channel(f'{host}:{port}', options=cfg.options)
        self._stub = DatasetsServiceStub(self._channel)

    def DatasetsList(self, request: DDatasetRequest) -> DDatasetResponse:
        return self._stub.DatasetsList(request)

    def GetDataset(self, request: DDatasetRequest) -> DDatasetResponse:
        return self._stub.GetDataset(request)

    def NewDataset(self, request: DDatasetRequest) -> DDatasetResponse:
        return self._stub.NewDataset(request)

    def DeleteDataset(self, request: DDatasetRequest) -> DDatasetResponse:
        return self._stub.DeleteDataset(request)

    def GetSample(self, request: DSampleRequest) -> DSampleResponse:
        return self._stub.GetSample(request)

    def NewSample(self, request: DSampleRequest) -> DSampleResponse:
        return self._stub.NewSample(request)

    def UpdateSample(self, request: DSampleRequest) -> DSampleResponse:
        return self._stub.UpdateSample(request)

    def GetItem(self, request: DItemRequest) -> DItemResponse:
        return self._stub.GetItem(request)

    def NewItem(self, request: DItemRequest) -> DItemResponse:
        return self._stub.NewItem(request)

    def UpdateItem(self, request: DItemRequest) -> DItemResponse:
        return self._stub.UpdateItem(request)


class DatasetsSimpleServiceClient(DatasetsServiceClient):

    def __init__(self, host='localhost', port=50051, cfg=DatasetsServiceClientCFG()):
        super(DatasetsSimpleServiceClient, self).__init__(host=host, port=port, cfg=cfg)

    def datasets_list(self, dataset_name: str = '') -> List[str]:
        """ Retrieves dataset names list

        :param dataset_name: query 'like' string for dataset name, defaults to ''
        :type dataset_name: str, optional
        :return: list of dataset name
        :rtype: List[str]
        """
        request = DDatasetRequest()
        request.dataset_name = dataset_name

        response = self.DatasetsList(request=request)
        results = []
        for dataset in response.datasets:
            results.append(dataset.name)

        return results

    def delete_dataset(self, dataset_name: str) -> bool:
        """ Deletes target dataset

        :param dataset_name: target dataset name
        :type dataset_name: str
        :return: TRUE if deletion is complete
        :rtype: bool
        """

        request = DDatasetRequest()
        request.dataset_name = dataset_name

        response = self.DeleteDataset(request)
        if response.status.code == 0:
            return True
        else:
            return False

    def new_dataset(self, dataset_name: str, dataset_category: str) -> Union[dict, None]:
        """ Creates new dataset

        :param dataset_name: target dataset name
        :type dataset_name: str
        :param dataset_category: target dataset category
        :type dataset_category: str
        :return: JSON-like dataset representation or None if errors occur
        :rtype: Union[dict, None]
        """

        request = DDatasetRequest()
        request.dataset_name = dataset_name
        request.dataset_category = dataset_category

        response = self.NewDataset(request)
        if len(response.datasets) > 0:
            return json_format.MessageToDict(response.datasets[0], including_default_value_fields=True, preserving_proto_field_name=True)
        else:
            return None

    def get_dataset(self, dataset_name: str, fetch_data: bool = False) -> Union[dict, None]:
        """ Retrieves single dataset as JSON-like dictionary

        :param dataset_name: target dataset name
        :type dataset_name: str
        :param fetch_data: TRUE to fetch also binary item data, defaults to False
        :type fetch_data: bool, optional
        :return: JSON-like dictionary representing the whole dataset
        :rtype: Union[dict, None]
        """

        request = DDatasetRequest()
        request.dataset_name = dataset_name
        request.fetch_data = fetch_data

        response = self.GetDataset(request)
        if len(response.datasets) > 0:
            return json_format.MessageToDict(response.datasets[0], including_default_value_fields=True, preserving_proto_field_name=True)
        else:
            return None

    def new_sample(self, dataset_name: str, metadata: dict = {}) -> Union[dict, None]:
        """ Creates new sample associated with target dataset

        :param dataset_name: target dataset name
        :type dataset_name: str
        :param metadata: sample associated metadata, defaults to {}
        :type metadata: dict, optional
        :return: JSON-like dictionary representing new sample or None if errors occur
        :rtype: Union[dict, None]
        """

        request = DSampleRequest()
        request.dataset_name = dataset_name
        json_format.ParseDict(metadata, request.metadata)

        response = self.NewSample(request)
        if len(response.samples) > 0:
            return json_format.MessageToDict(response.samples[0], including_default_value_fields=True, preserving_proto_field_name=True)
        else:
            return None

    def get_sample(self, dataset_name: str, sample_id: int, fetch_data: bool = False) -> Union[dict, None]:
        """ Retrieves single sample associated with target dataset

        :param dataset_name: target dataset name
        :type dataset_name: str
        :param sample_id: target sample id
        :type sample_id: int
        :param fetch_data: TRUE to fetch also binary item data, defaults to False
        :type fetch_data: bool, optional
        :return: JSON-like sample representation or None if errors occur
        :rtype: Union[dict, None]
        """

        request = DSampleRequest()
        request.dataset_name = dataset_name
        request.sample_id = sample_id
        request.fetch_data = fetch_data

        response = self.GetSample(request)
        if len(response.samples) > 0:
            return json_format.MessageToDict(response.samples[0], including_default_value_fields=True, preserving_proto_field_name=True)
        else:
            return None

    def update_sample(self, dataset_name: str, sample_id: int, metadata: dict, fetch_data: bool = False) -> Union[dict, None]:
        """ Retrieves single sample associated with target dataset

        :param dataset_name: target dataset name
        :type dataset_name: str
        :param sample_id: target sample id
        :type sample_id: int
        :param metadata: sample associated metadata, defaults to {}
        :type metadata: dict, optional
        :param fetch_data: TRUE to fetch also binary item data, defaults to False
        :type fetch_data: bool, optional
        :return: JSON-like sample representation or None if errors occur
        :rtype: Union[dict, None]
        """

        request = DSampleRequest()
        request.dataset_name = dataset_name
        request.sample_id = sample_id
        json_format.ParseDict(metadata, request.metadata)
        request.fetch_data = fetch_data

        response = self.UpdateSample(request)
        if len(response.samples) > 0:
            return json_format.MessageToDict(response.samples[0], including_default_value_fields=True, preserving_proto_field_name=True)
        else:
            return None

    def get_item(self, dataset_name: str, sample_id: int, item_name: str, fetch_data: bool = False) -> Union[dict, None]:
        """ Retrieves single item associated with target dataset/sample

        :param dataset_name: target dataset name
        :type dataset_name: str
        :param sample_id: target sample id
        :type sample_id: int
        :param item_name: target item name
        :type item_name: str
        :param fetch_data: TRUE to fetch also binary data, defaults to False
        :type fetch_data: bool, optional
        :return: JSON-like item representation or None if errors occur
        :rtype: Union[dict, None]
        """

        request = DItemRequest()
        request.dataset_name = dataset_name
        request.sample_id = sample_id
        request.item_name = item_name
        request.fetch_data = fetch_data

        response = self.GetItem(request)
        if len(response.items) > 0:
            return json_format.MessageToDict(response.items[0], including_default_value_fields=True, preserving_proto_field_name=True)
        else:
            return None

    def new_item(self, dataset_name: str, sample_id: int, item_name: str, data: bytes, data_encoding: str) -> Union[dict, None]:
        """ Creates new item associated with target dataset/sample

        :param dataset_name: target dataset name
        :type dataset_name: str
        :param sample_id: target sample id
        :type sample_id: int
        :param item_name: target item name
        :type item_name: str
        :param data: bytes to store
        :type data: bytes
        :param data_encoding: encoding for bytes to store
        :type data_encoding: str
        :return: JSON-like representation of created item or None if errors occur
        :rtype: Union[dict, None]
        """

        request = DItemRequest()
        request.dataset_name = dataset_name
        request.sample_id = sample_id
        request.item_name = item_name
        request.data = data
        request.data_encoding = data_encoding

        response = self.NewItem(request)
        if len(response.items) > 0:
            return json_format.MessageToDict(response.items[0], including_default_value_fields=True, preserving_proto_field_name=True)
        else:
            return None

    def update_item(self, dataset_name: str, sample_id: int, item_name: str, data: bytes, data_encoding: str) -> Union[dict, None]:
        """ Creates new item associated with target dataset/sample

        :param dataset_name: target dataset name
        :type dataset_name: str
        :param sample_id: target sample id
        :type sample_id: int
        :param item_name: target item name
        :type item_name: str
        :param data: bytes to store
        :type data: bytes
        :param data_encoding: encoding for bytes to store
        :type data_encoding: str
        :return: JSON-like representation of created item or None if errors occur
        :rtype: Union[dict, None]
        """

        request = DItemRequest()
        request.dataset_name = dataset_name
        request.sample_id = sample_id
        request.item_name = item_name
        request.data = data
        request.data_encoding = data_encoding

        response = self.UpdateItem(request)
        if len(response.items) > 0:
            return json_format.MessageToDict(response.items[0], including_default_value_fields=True, preserving_proto_field_name=True)
        else:
            return None

    def get_item_data(self, dataset_name: str, sample_id: int, item_name: str) -> Union[Tuple[bytes, str], None]:
        """ Retrives bytes data associated with target item

        :param dataset_name: target dataset name
        :type dataset_name: str
        :param sample_id: target sample id
        :type sample_id: int
        :param item_name: target item name
        :type item_name: str
        :return: tuple with (bytes, encoding) or None if errors occur
        :rtype: Union[Tuple[bytes, str], None]
        """

        request = DItemRequest()
        request.dataset_name = dataset_name
        request.sample_id = sample_id
        request.item_name = item_name
        request.fetch_data = True

        response = self.GetItem(request)
        if len(response.items) > 0:
            return response.items[0].data, response.items[0].data_encoding
        else:
            return None
