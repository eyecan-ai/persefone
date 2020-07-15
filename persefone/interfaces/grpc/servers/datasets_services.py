from pathlib import Path
from persefone.data.io.drivers.common import AbstractFileDriver
from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDataset, MongoDatasetsManager
from google.protobuf import json_format
from persefone.interfaces.proto.utils.comm import ResponseStatusUtils
from persefone.data.databases.mongo.model import MSample, MItem, MResource
from persefone.interfaces.proto.datasets_pb2 import DDataset, DItem, DSample
from persefone.interfaces.grpc.datasets_services_pb2_grpc import DatasetsServiceServicer, add_DatasetsServiceServicer_to_server
from persefone.interfaces.grpc.datasets_services_pb2 import (
    DDatasetRequest, DDatasetResponse,
    DSampleRequest, DSampleResponse,
    DItemRequest, DItemResponse
)
from typing import List
import grpc
from abc import ABC, abstractmethod


class DatasetsServiceCFG(object):
    DEFAULT_MAX_MESSAGE_LENGTH = -1

    def __init__(self):
        self.options = [
            ('grpc.max_send_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
        ]


class DatasetsService(ABC, DatasetsServiceServicer):

    def __init__(self):
        pass

    def register(self, grpc_server):
        add_DatasetsServiceServicer_to_server(self, grpc_server)

    @abstractmethod
    def DatasetsList(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:
        pass

    @abstractmethod
    def GetDataset(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:
        pass

    @abstractmethod
    def DeleteDataset(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:
        pass

    @abstractmethod
    def NewDataset(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:
        pass

    @abstractmethod
    def GetSample(self, request: DSampleRequest, context: grpc.ServicerContext) -> DSampleResponse:
        pass

    @abstractmethod
    def NewSample(self, request: DSampleRequest, context: grpc.ServicerContext) -> DSampleResponse:
        pass

    @abstractmethod
    def UpdateSample(self, request: DSampleRequest, context: grpc.ServicerContext) -> DSampleResponse:
        pass

    @abstractmethod
    def GetItem(self, request: DItemRequest, context: grpc.ServicerContext) -> DItemResponse:
        pass

    @abstractmethod
    def NewItem(self, request: DItemRequest, context: grpc.ServicerContext) -> DItemResponse:
        pass

    @abstractmethod
    def UpdateItem(self, request: DItemRequest, context: grpc.ServicerContext) -> DItemResponse:
        pass


class MongoDatasetService(DatasetsService):

    def __init__(self,
                 mongo_client: MongoDatabaseClient,
                 driver: AbstractFileDriver,
                 ):
        super(MongoDatasetService, self).__init__()

        self._mongo_client = mongo_client
        assert isinstance(driver, AbstractFileDriver), f"Invalid driver: {driver}"
        self._drivers = [driver]

    def _create_ddataset(self, mongo_dataset: MongoDataset) -> DDataset:
        ddataset = DDataset()
        ddataset.name = mongo_dataset.dataset.name
        ddataset.category = mongo_dataset.dataset.category.name
        return ddataset

    def _create_dsample(self, sample: MSample) -> DSample:
        dsample = DSample()
        json_format.ParseDict(sample.metadata, dsample.metadata)
        dsample.sample_id = sample.sample_id
        return dsample

    def _create_ditem(self, item: MItem) -> DItem:
        ditem = DItem()
        ditem.name = item.name
        return ditem

    def _get_dataset(self, request: DDatasetRequest):
        dataset_name = request.dataset_name
        return MongoDatasetsManager(self._mongo_client).get_dataset(
            dataset_name, drivers=self._drivers
        )

    def _create_dataset(self, request: DDatasetRequest):
        dataset_name = request.dataset_name
        dataset_category = request.dataset_category
        return MongoDatasetsManager(self._mongo_client).create_dataset(
            dataset_name, dataset_category, drivers=self._drivers
        )

    def _get_datasets(self, request: DDatasetRequest):
        dataset_name = request.dataset_name
        return MongoDatasetsManager(self._mongo_client).get_datasets(
            dataset_name=dataset_name, drivers=self._drivers
        )

    def _fetch_resource(self, mongo_dataset: MongoDataset, ditem: DItem, resources: List[MResource]) -> bool:
        for resource in resources:
            if resource.driver == self._drivers[0].driver_name():
                ditem.data = mongo_dataset.fetch_resource_to_blob(resource)
                ditem.data_encoding = Path(resource.uri).suffix
                ditem.has_data = True
                return True
        return False

    def DatasetsList(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:

        # Fetches list of MongoDataset s if any
        datasets = self._get_datasets(request)

        # Inits Response
        response = DDatasetResponse()

        # Build response status
        response.status.CopyFrom(ResponseStatusUtils.create_ok_status("all right!"))

        # Fills response with DDataset s if any
        for mongo_dataset in datasets:
            response.datasets.append(self._create_ddataset(mongo_dataset))

        return response

    def NewDataset(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:

        # Fetches list of MongoDataset s if any
        mongo_dataset: MongoDataset = self._create_dataset(request)

        # Inits Response
        response = DDatasetResponse()

        if mongo_dataset is None:  # No corresponding dataser found
            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Impossible to create: [{request.dataset_name}]"))
        else:
            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_ok_status())

            # Creates DDataset from MongoDataset
            ddataset = self._create_ddataset(mongo_dataset)

            # Fills Response with DDataset
            response.datasets.append(ddataset)

        return response

    def DeleteDataset(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:

        # Fetches list of MongoDataset s if any
        mongo_dataset: MongoDataset = self._get_dataset(request)

        # Inits Response
        response = DDatasetResponse()

        if mongo_dataset.dataset is None:  # No corresponding dataser found
            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Impossible to delete: [{ request.dataset_name}]"))
        else:

            if mongo_dataset.delete(security_name=request.dataset_name):

                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_ok_status())
            else:

                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Impossible to delete: [{ request.dataset_name}]"))

        return response

    def GetDataset(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:

        # Fetches MongoDataset  if any
        mongo_dataset: MongoDataset = self._get_dataset(request)

        # Inits Response
        response = DDatasetResponse()

        if mongo_dataset is None:  # No corresponding dataser found

            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No dataset found: [{request.dataset_name}]"))
        else:

            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_ok_status())

            # Creates DDataset from Mongo Dataset object
            ddataset = self._create_ddataset(mongo_dataset)

            # Fetches dataset samples
            samples = mongo_dataset.get_samples()
            for sample in samples:

                # Creates DSample from MSample
                dsample = self._create_dsample(sample)

                # Fetches sample items
                items = mongo_dataset.get_items(sample_idx=sample.sample_id)

                for item in items:

                    # Creates DItem from MItem
                    ditem = self._create_ditem(item)

                    # if fetch_data == True in request, fetches also tensor data in each DItem
                    if request.fetch_data:
                        self._fetch_resource(mongo_dataset, ditem, item.resources)

                    # Fills DSample with DItems
                    dsample.items.append(ditem)

                # Fills DDataset with DSample s
                ddataset.samples.append(dsample)

            # Fills Response with DDataset
            response.datasets.append(ddataset)

        return response

    def GetSample(self, request: DSampleRequest, context: grpc.ServicerContext) -> DSampleResponse:

        # Fetches MongoDataset  if any
        mongo_dataset: MongoDataset = self._get_dataset(request)

        # Inits Response
        response = DSampleResponse()

        if mongo_dataset is None:  # No corresponding dataser found

            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No dataset found: [{request.dataset_name}]"))
        else:

            sample = mongo_dataset.get_sample(request.sample_id)

            if sample is None:

                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No sample found with id: [{request.sample_id}]"))

            else:
                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_ok_status())

                # Creates DSample from MSample
                dsample = self._create_dsample(sample)

                # Fetches sample items
                items = mongo_dataset.get_items(sample_idx=sample.sample_id)

                for item in items:

                    # Creates DItem from MItem
                    ditem = self._create_ditem(item)

                    # if fetch_data == True in request, fetches also tensor data in each DItem
                    if request.fetch_data:
                        self._fetch_resource(mongo_dataset, ditem, item.resources)

                    # Fills DSample with DItems
                    dsample.items.append(ditem)

                response.samples.append(dsample)

        return response

    def NewSample(self, request: DSampleRequest, context: grpc.ServicerContext) -> DSampleResponse:

        # Fetches MongoDataset  if any
        mongo_dataset: MongoDataset = self._get_dataset(request)

        # Inits Response
        response = DSampleResponse()

        if mongo_dataset is None:  # No corresponding dataser found

            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No dataset found: [{request.dataset_name}]"))
        else:

            metadata = json_format.MessageToDict(request.metadata)

            sample = mongo_dataset.add_sample(metadata=metadata)

            if sample is None:

                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Failed creating sample"))

            else:
                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_ok_status())

                # Creates DSample from MSample
                dsample = self._create_dsample(sample)
                response.samples.append(dsample)

        return response

    def UpdateSample(self, request: DSampleRequest, context: grpc.ServicerContext) -> DSampleResponse:

        # Fetches MongoDataset  if any
        mongo_dataset: MongoDataset = self._get_dataset(request)

        # Inits Response
        response = DSampleResponse()

        if mongo_dataset is None:  # No corresponding dataser found

            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No dataset found: [{request.dataset_name}]"))
        else:

            sample = mongo_dataset.get_sample(request.sample_id)

            if sample is None:

                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No sample found with id: [{request.sample_id}]"))

            else:
                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_ok_status())

                # Update metdata
                sample.metadata = json_format.MessageToDict(request.metadata)
                sample.save()

                # Creates DSample from MSample
                dsample = self._create_dsample(sample)

                # Fetches sample items
                items = mongo_dataset.get_items(sample_idx=sample.sample_id)

                for item in items:

                    # Creates DItem from MItem
                    ditem = self._create_ditem(item)

                    # if fetch_data == True in request, fetches also tensor data in each DItem
                    if request.fetch_data:
                        self._fetch_resource(mongo_dataset, ditem, item.resources)

                    # Fills DSample with DItems
                    dsample.items.append(ditem)

                response.samples.append(dsample)

        return response

    def GetItem(self, request: DItemRequest, context: grpc.ServicerContext) -> DItemResponse:

        # Fetches MongoDataset  if any
        mongo_dataset: MongoDataset = self._get_dataset(request)

        # Inits Response
        response = DItemResponse()

        if mongo_dataset is None:  # No corresponding dataser found

            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No dataset found: [{ request.dataset_name}]"))
        else:

            sample = mongo_dataset.get_sample(request.sample_id)

            if sample is None:

                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No sample found with id: [{request.sample_id}]"))

            else:

                item = mongo_dataset.get_item(sample.sample_id, request.item_name)

                if item is None:

                    # Build response status
                    response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No item found with name: [{request.item_name}]"))

                else:
                    # Build response status
                    response.status.CopyFrom(ResponseStatusUtils.create_ok_status())

                    ditem = self._create_ditem(item)

                    # if fetch_data == True in request, fetches also tensor data in each DItem
                    if request.fetch_data:
                        self._fetch_resource(mongo_dataset, ditem, item.resources)

                    response.items.append(ditem)

        return response

    def NewItem(self, request: DItemRequest, context: grpc.ServicerContext) -> DItemResponse:

        # Fetches MongoDataset  if any
        mongo_dataset: MongoDataset = self._get_dataset(request)

        # Inits Response
        response = DItemResponse()

        if mongo_dataset is None:  # No corresponding dataser found

            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No dataset found: [{request.dataset_name}]"))
        else:

            sample = mongo_dataset.get_sample(request.sample_id)

            if sample is None:

                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No sample found with id: [{request.sample_id}]"))

            else:

                item = mongo_dataset.add_item(sample.sample_id, request.item_name)

                if item is None:

                    # Build response status
                    response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Item with the same name found!"))

                else:

                    # Build response status
                    response.status.CopyFrom(ResponseStatusUtils.create_ok_status())

                    resource = mongo_dataset.push_resource_from_blob(
                        sample.sample_id,
                        item.name,
                        item.name,
                        request.data,
                        request.data_encoding,
                        self._drivers[0].driver_name()
                    )

                    if resource is None:
                        # Build response status
                        response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Failed to create Resource"))
                    else:

                        ditem = self._create_ditem(item)
                        response.items.append(ditem)

        return response

    def UpdateItem(self, request: DItemRequest, context: grpc.ServicerContext) -> DItemResponse:

        # Fetches MongoDataset  if any
        mongo_dataset: MongoDataset = self._get_dataset(request)

        # Inits Response
        response = DItemResponse()

        if mongo_dataset is None:  # No corresponding dataser found

            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No dataset found: [{ request.dataset_name}]"))
        else:

            sample = mongo_dataset.get_sample(request.sample_id)

            if sample is None:

                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No sample found with id: [{request.sample_id}]"))

            else:

                item = mongo_dataset.get_item(sample.sample_id, request.item_name)

                if item is None:

                    # Build response status
                    response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No item found with name: [{request.item_name}]"))

                else:

                    # Build response status
                    response.status.CopyFrom(ResponseStatusUtils.create_ok_status())

                    resource = mongo_dataset.push_resource_from_blob(
                        sample.sample_id,
                        item.name,
                        item.name,
                        request.data,
                        request.data_encoding,
                        self._drivers[0].driver_name()
                    )

                    if resource is None:
                        # Build response status
                        response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Failed to create Resource"))
                    else:
                        ditem = self._create_ditem(item)
                        response.items.append(ditem)

        return response
