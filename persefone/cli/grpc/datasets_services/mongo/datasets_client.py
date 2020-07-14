import sys
from persefone.interfaces.grpc.datasets_services_pb2 import (
    DDatasetRequest, DDatasetResponse,
    DSampleRequest, DSampleResponse,
    DItemRequest, DItemResponse
)
from persefone.interfaces.proto.utils.dtensor import DTensorUtils
from persefone.interfaces.grpc.clients.datasets_services import DatasetsServiceClient, DatasetsServiceClientCFG, DatasetsSimpleServiceClient
import grpc
from google.protobuf import json_format
import cv2
from io import BytesIO
import imageio
import pprint
from persefone.utils.bytes import DataCoding

client = DatasetsSimpleServiceClient()

print(client.dataset_list())

print(client.get_dataset('green'))

res = client.get_sample('green', 10, fetch_data=False)
pprint.pprint(res)

res = client.get_item('green', 10, 'image', fetch_data=True)
pprint.pprint(res)

data, encoding = client.get_item_data('green', 10, 'image')

# image = DataCoding.bytes_to_data(data, encoding)
# cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)

sample = client.new_sample('green', metadata={'a': 2.2, 'b': 'ciao', 'd': [0, 1, 2, 3]})
print(sample)


data, data_encoding = DataCoding.file_to_bytes(filename='/home/daniele/Downloads/image.bmp')
item = client.new_item('green', sample['sample_id'], 'image', data, data_encoding)
print(sample, item)


r_data, r_encoding = client.get_item_data('green', sample['sample_id'], 'image')
r = DataCoding.bytes_to_data(r_data, r_encoding)

cv2.imshow("image", r)
cv2.waitKey(0)
print(r.shape)
# print(client.delete_dataset('green'))

sys.exit(0)
# request = DDatasetRequest()
client = DatasetsServiceClient()
print(client)
# request.dataset_name = 'green'
#request.fetch_data = True


# response: DDatasetResponse = client.DatasetsList(request=request)
# print(response)
# print(response.status.message)

#response: DDatasetResponse = client.GetDataset(request=request)


# print(json_format.MessageToJson(response))
# print(response.ByteSize())


# NEW SAMPLE
# sample_request = DSampleRequest()
# sample_request.dataset_name = 'green'
# json_format.ParseDict({'a': 2.2, 'b': 'ciao'}, sample_request.metadata)

# sample_response: DSampleResponse = client.NewSample(sample_request)
# print(json_format.MessageToJson(sample_response))
# created_id = sample_response.samples[0].sample_id

# GET SAMPLE
# sample_request = DSampleRequest()
# sample_request.dataset_name = 'green'
# sample_request.sample_id = 35

# sample_response: DSampleResponse = client.GetSample(sample_request)
# print(json_format.MessageToJson(sample_response))
# print(sample_response.samples[0].sample_id)

if True:

    sample_id = 0
    while True:
        # GET ITEM
        item_request = DItemRequest()
        item_request.dataset_name = 'green'
        item_request.sample_id = sample_id
        item_request.item_name = 'image'
        item_request.fetch_data = True

        item_response: DItemResponse = client.GetItem(item_request)
        if item_response.status.code == 0:
            item = item_response.items[0]

            encoding = item.data_encoding if len(item.data_encoding) > 0 else 'jpg'
            buff = BytesIO(item.data)
            image = imageio.imread(buff.getbuffer(), format=encoding)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("image", image)
            cv2.waitKey(0)
            print(image.shape, image.dtype)
            sample_id += 1
        else:
            break

else:

    sample_request = DSampleRequest()
    sample_request.dataset_name = 'green'
    json_format.ParseDict({'a': 2.2, 'b': 'ciao'}, sample_request.metadata)

    sample_response: DSampleResponse = client.NewSample(sample_request)
    print(json_format.MessageToJson(sample_response))
    sample_id = sample_response.samples[0].sample_id

    for name in ['a', 'b', 'c', 'd']:
        item_request = DItemRequest()
        item_request.dataset_name = 'green'
        item_request.sample_id = sample_id
        item_request.item_name = name

        image = open('/home/daniele/Downloads/cooltext-357159549532765.png', 'rb').read()
        item_request.data = image
        item_request.data_extension = '.png'

        item_response: DItemResponse = client.NewItem(item_request)
        print(item_response)
