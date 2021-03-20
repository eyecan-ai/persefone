

from persefone.utils.datasets.fake import FakeDatasetGenerator
import cv2
import click
from pathlib import Path
import yaml
import json


@click.command("Generate Fake Dataset")
@click.option('--output_folder', type=str, required=True, help="Output folder")
@click.option('--size', type=int, required=True, help="Number of output samples")
@click.option('--image_size', type=int, default=256, help="Output image size")
def generate_fake_dataset(output_folder, size, image_size):

    zfill = 5
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    generator = FakeDatasetGenerator()

    for idx in range(size):

        # Generate sample
        sample = generator.generate_image_sample([image_size, image_size])

        # Extracts metadata
        metadata = {
            'bboxes': sample['bboxes'],
            'keypoints': sample['keypoints'],
            'label': sample['label'],
            'id': sample['id']
        }

        metadata = json.loads(json.dumps(metadata))

        # Naming
        name = str(idx).zfill(zfill)
        image_name = f'{name}_image.jpg'
        mask_name = f'{name}_mask.png'
        instances_name = f'{name}_inst.png'
        metadata_name = f'{name}_metadata.yml'
        cv2.imwrite(str(output_folder / image_name), sample['rgb'])
        cv2.imwrite(str(output_folder / mask_name), sample['mask'])
        cv2.imwrite(str(output_folder / instances_name), sample['instances'])

        yaml.safe_dump(metadata, open(str(output_folder / metadata_name), 'w'))


if __name__ == "__main__":
    generate_fake_dataset()
