from persefone.interfaces.grpc.clients.tensor_services import MetaImagesServiceClient
import grpc
import logging
import click
from persefone.utils.cli_options import cli_base_options, cli_host_options
import cv2
import time


@click.command("Launch MetaImages Client")
@cli_base_options
@cli_host_options
@click.option("--target_image", required=True, help="Image to send filename")
@click.option("--show_images", default=False, is_flag=True, help="Show image before/after send")
@click.option("--repetitions", default=1, type=int, help="Repeat send times")
@click.option("--transpose_image", default=True, type=bool, help="Does server transpose image?")
@click.option("--rescale_image", default=0.5, type=float, help="Rescale percentage applyied by server? (< 0 to disable)")
def metaimages_client(**options):

    print(options)
    show_images = options.get('show_images')
    logging.basicConfig(level=logging.DEBUG if options.get('debug') else logging.WARNING)

    try:

        # Creates client
        client = MetaImagesServiceClient(
            host=options.get('host'),
            port=options.get('port')
        )

        for _ in range(options.get('repetitions')):

            # input image
            image = cv2.imread(options.get('target_image'))

            if show_images:  # debug show
                cv2.imshow("image", image)
                cv2.waitKey(0)

            t1 = time.time()

            # Metadata to send
            metadata = {
                'transpose': options.get('transpose_image'),
                'scale_percent': options.get('rescale_image')
            }

            # GRPC call
            reply_images, reply_action = client.send_meta_images([image], metadata)

            logging.debug(f"Round trip time: {time.time() - t1} seconds")

            if show_images:  # debug show
                cv2.imshow("image", reply_images[0])
                cv2.waitKey(0)

            logging.debug(f"Reply metadata: {reply_action}")

    except grpc.RpcError as e:
        logging.error(e)
        print(f"Failed to connect to: {options.get('host')}:{options.get('port')}")


if __name__ == "__main__":
    metaimages_client()
