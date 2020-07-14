
from persefone.interfaces.proto.datasets_pb2 import DDataset, DItem, DSample
from persefone.data.databases.mongo.model import MSample, MItem
from persefone.interfaces.grpc.datasets_services_pb2 import (
    DDatasetRequest, DDatasetResponse,
    DSampleRequest, DSampleResponse,
    DItemRequest, DItemResponse
)
from persefone.interfaces.grpc.servers.datasets_services import DatasetsServiceServer, DatasetsServiceServerCFG
from persefone.interfaces.proto.utils.comm import ResponseStatusUtils
from persefone.interfaces.proto.utils.dtensor import DTensorUtils
from google.protobuf import json_format
import grpc
import time
from typing import List
from persefone.data.databases.mongo.repositories import DatasetsRepository
from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDataset
from persefone.data.io.drivers.safefs import SafeFilesystemDriver
from pathlib import Path

driver = SafeFilesystemDriver.create_from_configuration_file('securefs_driver.yml')
mongo_client = MongoDatabaseClient.create_from_configuration_file(filename='database.yml')

mongo_client.connect()


class MyDatasetServer(DatasetsServiceServer):

    def __init__(self, host='0.0.0.0', port=50051, max_workers=1, options: DatasetsServiceServerCFG = DatasetsServiceServerCFG().options):
        super(MyDatasetServer, self).__init__(host=host, port=port, max_workers=max_workers, options=options)

    def create_ddataset(self, mongo_dataset: MongoDataset) -> DDataset:
        ddataset = DDataset()
        ddataset.name = mongo_dataset.dataset.name
        ddataset.category = mongo_dataset.dataset.category.name
        return ddataset

    def create_dsample(self, sample: MSample) -> DSample:
        dsample = DSample()
        json_format.ParseDict(sample.metadata, dsample.metadata)
        dsample.sample_id = sample.sample_id
        return dsample

    def create_ditem(self, item: MItem) -> DItem:
        ditem = DItem()
        ditem.name = item.name
        return ditem

    def DatasetsList(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:

        # Fetches list of MongoDataset s if any
        dataset_name = request.dataset_name
        datasets = mongo_client.get_datasets(dataset_name, drivers=[driver])

        # Inits Response
        response = DDatasetResponse()

        # Build response status
        response.status.CopyFrom(ResponseStatusUtils.create_ok_status("all right!"))

        # Fills response with DDataset s if any
        for mongo_dataset in datasets:
            response.datasets.append(self.create_ddataset(mongo_dataset))

        return response

    def NewDataset(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:

        # Fetches list of MongoDataset s if any
        dataset_name = request.dataset_name
        dataset_category = request.dataset_category

        mongo_dataset: MongoDataset = mongo_client.create_dataset(dataset_name, dataset_category, drivers=[driver])

        # Inits Response
        response = DDatasetResponse()

        if mongo_dataset is None:  # No corresponding dataser found
            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Impossible to create: [{dataset_name}]"))
        else:
            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_ok_status())

            # Creates DDataset from MongoDataset
            ddataset = self.create_ddataset(mongo_dataset)

            # Fills Response with DDataset
            response.datasets.append(ddataset)

        return response

    def DeleteDataset(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:

        # Fetches list of MongoDataset s if any
        dataset_name = request.dataset_name

        mongo_dataset: MongoDataset = mongo_client.get_dataset(dataset_name=dataset_name, drivers=[driver])

        # Inits Response
        response = DDatasetResponse()

        if mongo_dataset is None:  # No corresponding dataser found
            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Impossible to delete: [{dataset_name}]"))
        else:

            if mongo_dataset.delete(security_name=dataset_name):

                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_ok_status())
            else:

                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Impossible to delete: [{dataset_name}]"))

        return response

    def GetDataset(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:

        # Fetches MongoDataset  if any
        dataset_name = request.dataset_name
        mongo_dataset: MongoDataset = mongo_client.get_dataset(dataset_name=dataset_name, drivers=[driver])

        # Inits Response
        response = DDatasetResponse()

        if mongo_dataset is None:  # No corresponding dataser found

            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No dataset found: [{dataset_name}]"))
        else:

            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_ok_status())

            # Creates DDataset from Mongo Dataset object
            ddataset = self.create_ddataset(mongo_dataset)

            # Fetches dataset samples
            samples = mongo_dataset.get_samples()
            for sample in samples:

                # Creates DSample from MSample
                dsample = self.create_dsample(sample)

                # Fetches sample items
                items = mongo_dataset.get_items(sample_idx=sample.sample_id)

                for item in items:

                    # Creates DItem from MItem
                    ditem = self.create_ditem(item)

                    # if fetch_data == True in request, fetches also tensor data in each DItem
                    if request.fetch_data:
                        for resource in item.resources:
                            ditem.data = mongo_dataset.fetch_resource_to_blob(resource)
                            ditem.data_encoding = Path(resource.uri).suffix

                    # Fills DSample with DItems
                    dsample.items.append(ditem)

                # Fills DDataset with DSample s
                ddataset.samples.append(dsample)

            # Fills Response with DDataset
            response.datasets.append(ddataset)

        return response

    def GetSample(self, request: DSampleRequest, context: grpc.ServicerContext) -> DSampleResponse:

        # Fetches MongoDataset  if any
        dataset_name = request.dataset_name
        mongo_dataset: MongoDataset = mongo_client.get_dataset(dataset_name=dataset_name, drivers=[driver])

        # Inits Response
        response = DSampleResponse()

        if mongo_dataset is None:  # No corresponding dataser found

            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No dataset found: [{dataset_name}]"))
        else:

            sample = mongo_dataset.get_sample(request.sample_id)

            if sample is None:

                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No sample found with id: [{request.sample_id}]"))

            else:
                # Build response status
                response.status.CopyFrom(ResponseStatusUtils.create_ok_status())

                # Creates DSample from MSample
                dsample = self.create_dsample(sample)

                # Fetches sample items
                items = mongo_dataset.get_items(sample_idx=sample.sample_id)

                for item in items:

                    # Creates DItem from MItem
                    ditem = self.create_ditem(item)

                    # if fetch_data == True in request, fetches also tensor data in each DItem
                    if request.fetch_data:
                        for resource in item.resources:
                            ditem.data = mongo_dataset.fetch_resource_to_blob(resource)
                            ditem.data_encoding = Path(resource.uri).suffix

                    # Fills DSample with DItems
                    dsample.items.append(ditem)

                response.samples.append(dsample)

        return response

    def NewSample(self, request: DSampleRequest, context: grpc.ServicerContext) -> DSampleResponse:

        # Fetches MongoDataset  if any
        dataset_name = request.dataset_name
        mongo_dataset: MongoDataset = mongo_client.get_dataset(dataset_name=dataset_name, drivers=[driver])

        # Inits Response
        response = DSampleResponse()

        if mongo_dataset is None:  # No corresponding dataser found

            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No dataset found: [{dataset_name}]"))
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
                dsample = self.create_dsample(sample)
                response.samples.append(dsample)

        return response

    def GetItem(self, request: DItemRequest, context: grpc.ServicerContext) -> DItemResponse:

        # Fetches MongoDataset  if any
        dataset_name = request.dataset_name
        mongo_dataset: MongoDataset = mongo_client.get_dataset(dataset_name=dataset_name, drivers=[driver])

        # Inits Response
        response = DItemResponse()

        if mongo_dataset is None:  # No corresponding dataser found

            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No dataset found: [{dataset_name}]"))
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

                    ditem = self.create_ditem(item)

                    # if fetch_data == True in request, fetches also tensor data in each DItem
                    if request.fetch_data:
                        for resource in item.resources:
                            ditem.data = mongo_dataset.fetch_resource_to_blob(resource)
                            ditem.data_encoding = Path(resource.uri).suffix

                    response.items.append(ditem)

        return response

    def NewItem(self, request: DItemRequest, context: grpc.ServicerContext) -> DItemResponse:

        # Fetches MongoDataset  if any
        dataset_name = request.dataset_name
        mongo_dataset: MongoDataset = mongo_client.get_dataset(dataset_name=dataset_name, drivers=[driver])

        # Inits Response
        response = DItemResponse()

        if mongo_dataset is None:  # No corresponding dataser found

            # Build response status
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"No dataset found: [{dataset_name}]"))
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
                        driver.driver_name()
                    )

                    if resource is None:
                        # Build response status
                        response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Failed to create Resource"))
                    else:

                        ditem = self.create_ditem(item)
                        response.items.append(ditem)

        return response


server = MyDatasetServer()
server.start()
server.wait_for_termination()
