from persefone.interfaces.grpc.clients.tensor_services import MetaImagesServiceClient
from persefone.interfaces.proto.data_pb2 import DTensorBundle
import grpc
import logging
import numpy as np
import click
from persefone.utils.cli_options import cli_base_options, cli_host_options


@click.command("Launch MetaImages Client")
@cli_base_options
@cli_host_options
def metaimages_client(**options):

    logging.basicConfig(level=logging.DEBUG if options.get('debug') else logging.WARNING)

    try:
        client = MetaImagesServiceClient(
            host=options.get('host'),
            port=options.get('port')
        )

        reply_images, reply_action = client.send_meta_images([np.random.uniform(0, 1, (3, 255, 255))], {'a': 22.2})
        print(reply_action)

    except grpc.RpcError as e:
        logging.error(e)
        print(f"Failed to connect to: {options.get('host')}:{options.get('port')}")


if __name__ == "__main__":
    metaimages_client()
