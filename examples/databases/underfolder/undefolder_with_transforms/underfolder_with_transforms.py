import numpy as np
import cv2
import click
from persefone.data.databases.filesystem.underfolder import (
    TransformedUnderfolderDatabase
)


@click.command("Show Underfolder with Transforms")
@click.option("--dataset_folder", required=True)
@click.option("--augmentations_file", required=True)
def show_dataset_with_transforms(dataset_folder, augmentations_file):

    t_database = TransformedUnderfolderDatabase(folder=dataset_folder, augmentations_file=augmentations_file)

    for sample in t_database:
        for key in sample:
            data = sample[key]
            if isinstance(data, np.ndarray):
                cv2.imshow(key, data)

        if cv2.waitKey(0) == ord('q'):
            return


if __name__ == "__main__":
    show_dataset_with_transforms()
