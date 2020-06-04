import pytest
import numpy as np
from persefone.utils.datasets.fake import FakeDatasetGenerator


@pytest.mark.utils_datasets_fake
class TestFakeDatasetGenerator(object):

    def test_generation(self):

        generator = FakeDatasetGenerator()
        for size in ([256, 256], [512, 512], [800, 600], [400, 1000]):

            # Random background
            channels = 3
            bg = generator.generate_random_background(size, channels=channels)
            assert bg.shape == (size[1], size[0], channels), "Background size is wrong!"

            # Objects 2D
            obj = generator.generate_random_object_2D(size)
            print(obj)

            fields = ['center', 'size', 'tl', 'br', 'w', 'h', 'label']
            for field in fields:
                assert field in obj, f"Missing field '{field}' in object."
                assert isinstance(obj[field], np.ndarray) or isinstance(obj[field], int) or isinstance(obj[field], float)

            # Object 2D Boundingbox
            box = generator.generate_2d_object_bbox(size, obj)
            assert len(box) == 5, "Bounding Box fields are wrong"

            # Object 2D Keypoints
            KP_N = 3
            keypoints = generator.generate_2d_object_keypoints(size, obj)
            assert len(keypoints) == KP_N, "Number of keypoints is wrong!"
            for i in range(KP_N):
                assert len(keypoints[i]) == 6, "Keypoints fields are wrong"

            # Generate Data
            OBJ_N = 7
            LABELS = 5
            data = generator.generate_image_sample(size, max_label=LABELS, objects_number_range=[OBJ_N, OBJ_N + 1])

            assert data['rgb'].shape == (size[1], size[0], 3), "RGB shape is wrong"
            assert data['mask'].shape == (size[1], size[0]), "Mask shape is wrong"
            assert data['instances'].shape == (size[1], size[0]), "Instances shape is wrong"
            assert len(np.unique(data['instances'])) <= OBJ_N + 1, "Instances number is wrong"

            assert len(data['bboxes']) == OBJ_N, "BBoxes shape is wrong"
            assert len(data['bboxes'][0]) == 5, "Box shape is wrong"
            assert len(data['keypoints']) == OBJ_N * 3, "Keypoints shape is wrong"
            assert len(data['keypoints'][0]) == 6, " Keypoint shape is wrong"

            # Image.fromarray(data['rgb']).save('/tmp/rgb.png')
            # Image.fromarray((255. * data['mask'] / data['mask'].max()).astype(np.uint8)).save('/tmp/mask.png')
            # Image.fromarray((255. * data['instances'] / data['instances'].max()).astype(np.uint8)).save('/tmp/instances.png')
