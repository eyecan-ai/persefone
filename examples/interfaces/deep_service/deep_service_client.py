
from persefone.interfaces.grpc.clients.deep_services import DeepServicePack, SimpleDeepServiceClient

# Creates Deep Service client
client = SimpleDeepServiceClient()

# Create Pack for ping message
pack = DeepServicePack()
pack.metadata = {'_schema': 'ping', 'time': 3}
reply_pack = client.deep_serve(pack)
print(reply_pack.__dict__)

# Create Pack for action message
pack = DeepServicePack()
pack.metadata = {'_schema': 'action', 'command': 'new_command'}
reply_pack = client.deep_serve(pack)
print(reply_pack.__dict__)

# Create Pack for invalid message
pack = DeepServicePack()
pack.metadata = {'_schema': 'INVALID'}
reply_pack = client.deep_serve(pack)
print(reply_pack.__dict__)
