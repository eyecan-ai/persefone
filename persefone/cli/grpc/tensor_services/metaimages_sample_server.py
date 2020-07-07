from persefone.interfaces.grpc.servers.tensor_services import MetaImagesTensorServer
from persefone.interfaces.proto.data_pb2 import DTensorBundle
import grpc
import logging
import click
from persefone.utils.cli_options import cli_base_options, cli_host_options
import cv2


@click.command("Launch MetaImages Server")
@cli_base_options
@cli_host_options
def metaimages_server(**options):

    logging.basicConfig(level=logging.DEBUG if options.get('debug') else logging.WARNING)

    # Custom user callback receiving list of images with metadata
    def tensor_callback(images, metadata):
        new_images = []
        for image in images:

            new_image = image.copy()

            # apply rescale
            scale_percent = metadata.get('scale_percent', 0.5)
            if scale_percent > 0:
                width = int(image.shape[1] * scale_percent)
                height = int(image.shape[0] * scale_percent)
                dim = (width, height)
                new_image = cv2.resize(new_image, dim, interpolation=cv2.INTER_AREA)

            # Apply transpose
            transpose = metadata.get('transpose', True)
            if transpose:
                new_image = cv2.transpose(new_image)

            new_images.append(new_image)

        return new_images, {
            'status': 'ok',
            'transformed_images': len(new_images),
            'scale_percent': scale_percent,
            'transpose': transpose
        }

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
