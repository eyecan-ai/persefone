import numpy as np
from PIL import Image, ImageDraw
from persefone.utils.colors.palettes import MaterialPalette, Palette
import uuid


class FakeDatasetGenerator(object):

    def __init__(self, palette=None):
        self.__palette = palette if palette is not None else MaterialPalette()
        pass

    @property
    def palette(self) -> Palette:
        return self.__palette

    def generate_random_background(self, size, channels=3):
        return np.random.uniform(0, 255, (size[1], size[0], channels)).astype(np.uint8)

    def generate_random_object_2D(self, size, max_label=5):
        size = np.array(size)
        center = np.random.uniform(0.1 * size[0], 0.9 * size[0], (2,))
        random_size = np.random.uniform(size / 10, size / 2, (2,))
        top_left = np.array([center[0] - random_size[0] * 0.5, center[1] - random_size[1] * 0.5])
        bottom_right = np.array([center[0] + random_size[0] * 0.5, center[1] + random_size[1] * 0.5])
        width = random_size[0]
        height = random_size[1]
        label = np.random.randint(0, max_label + 1)

        return {
            'center': center,
            'size': random_size,
            'tl': top_left,
            'br': bottom_right,
            'w': width,
            'h': height,
            'label': label
        }

    def generate_2d_object_bbox(self, size, obj):
        center = obj['center']
        w, h = size
        return (center[0] / w, center[1] / h, obj['w'] / w, obj['h'] / h, obj['label'])

    def generate_2d_object_keypoints(self, size, obj):

        center = obj['center']
        tl = obj['tl']
        br = obj['br']
        obj_size = obj['size']
        w, h = size
        label = obj['label']

        orientation = 1 if obj_size[0] > obj_size[1] else 0
        scale = obj_size[0] / w

        kp0 = (center[0] / w, center[1] / h, scale, orientation, label, 0)
        kp1 = (tl[0] / w, tl[1] / h, scale, orientation, label, 1)
        kp2 = (br[0] / w, br[1] / h, scale, orientation, label, 2)

        return [kp0, kp1, kp2]

    def generate_2d_objects_images(self, size, objects):

        image = Image.fromarray(self.generate_random_background(size))
        mask = Image.new('L', size)
        instances = Image.new('L', size)

        for index, obj in enumerate(objects):
            label = obj['label']
            coords = tuple(obj['tl']), tuple(obj['br'])
            color = self.palette.get_color(label).rgb

            ImageDraw.Draw(image).ellipse(coords, fill=color)
            ImageDraw.Draw(mask).ellipse(coords, fill=label + 1)
            ImageDraw.Draw(instances).ellipse(coords, fill=index + 1)

        return {
            'rgb': np.array(image),
            'mask': np.array(mask),
            'instances': np.array(instances)
        }

    def generate_image_sample(self, size, max_label=5, objects_number_range=[1, 5]):

        objects = []
        bboxes = []
        keypoints = []
        objects_number = np.random.randint(objects_number_range[0], objects_number_range[1])
        for n in range(objects_number):
            obj = self.generate_random_object_2D(size, max_label=max_label)
            box = self.generate_2d_object_bbox(size, obj)
            kps = self.generate_2d_object_keypoints(size, obj)
            objects.append(obj)
            bboxes.append(box)
            keypoints.extend(kps)

        data = self.generate_2d_objects_images(size, objects)
        data.update({
            'bboxes': bboxes,
            'keypoints': keypoints,
            'label': np.random.randint(max_label + 1),
            'id': str(uuid.uuid1())
        })
        return data
