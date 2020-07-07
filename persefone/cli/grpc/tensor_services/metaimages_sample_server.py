from persefone.interfaces.grpc.servers.tensor_services import MetaImagesTensorServer
from persefone.interfaces.proto.data_pb2 import DTensorBundle
import grpc
import logging
import click
from persefone.utils.cli_options import cli_base_options, cli_host_options


@click.command("Launch MetaImages Server")
@cli_base_options
@cli_host_options
def metaimages_server(**options):

    logging.basicConfig(level=logging.DEBUG if options.get('debug') else logging.WARNING)

    # Custom user callback receiving list of images with metadata
    def tensor_callback(images, metadata):
        new_images = []
        for image in images:
            new_images.append(image.T)

        return new_images, {'status': 'ok', 'transposed_images': len(new_images)}

    # Creates New MetaImages Server
    server = MetaImagesTensorServer(
        user_callback=tensor_callback,
        host=options.get('host'),
        port=options.get('port')
    )

    # Starts&Wait
    server.start()
    logging.info("Server started ...")
    server.wait_for_termination()


if __name__ == "__main__":
    metaimages_server()
