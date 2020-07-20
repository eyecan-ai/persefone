
from persefone.interfaces.proto.utils.dtensor import DTensorUtils
from persefone.utils.bytes import DataCoding
import numpy as np
import pytest


class TestDTensorUtils(object):

    @pytest.fixture()
    def testing_arrays(self):
        return [
            {'size': 0, 'shape': (0,)},
            {'size': 100, 'shape': (25, 2, 2, 1, 1)},
            {'size': 1, 'shape': (1,)},
            {'size': 1, 'shape': (1, 1, 1, 1, 1)},
            {'size': 16, 'shape': (4, 4)},
            {'size': 3 * 600 * 800, 'shape': (3, 600, 800)},
        ]

    def test_numpy_conversion(self, testing_arrays):

        for test_array in testing_arrays:
            for numpy_dtype, _ in DTensorUtils.NUMPY_TYPES_MAPPING.items():
                x = np.array(np.random.uniform(-100, 100, test_array['shape']), dtype=np.dtype(numpy_dtype))

                dtensor = DTensorUtils.numpy_to_dtensor(x)
                x_back = DTensorUtils.dtensor_to_numpy(dtensor)

                print(x.shape, x.dtype)
                assert x.shape == x_back.shape, "Arrays shapes must be equal after conversion"
                assert x.dtype == x_back.dtype, "Arrays shapes must be equal after conversion"
                assert np.array_equal(x, x_back), "Arrays contents must be equal after conversion"

    def test_numpy_conversion_into_bundle(self, testing_arrays):

        numpy_arrays = []
        for test_array in testing_arrays:
            for numpy_dtype, _ in DTensorUtils.NUMPY_TYPES_MAPPING.items():
                x = np.array(np.random.uniform(-100, 100, test_array['shape']), dtype=np.dtype(numpy_dtype))
                numpy_arrays.append(x)

        action_command = "process_bundle"
        bundle = DTensorUtils.numpy_to_dtensor_bundle(numpy_arrays, action_command)

        retrieved_arrays, retrieved_command = DTensorUtils.dtensor_bundle_to_numpy(bundle)

        assert action_command == retrieved_command, f"Wrong retrieved command: '{retrieved_command}'"
        assert len(retrieved_arrays) == len(numpy_arrays), f"Number of retrieved arrays is wrong! {len(retrieved_arrays)}/{len(numpy_arrays)}"
        for idx in range(len(retrieved_arrays)):
            print("Comparing DTensors arrays:", numpy_arrays[idx].shape, retrieved_arrays[idx].shape)
            a, b = numpy_arrays[idx], retrieved_arrays[idx]
            assert a.shape == b.shape, f"Retrieved array ({idx}),  wrong shape!"
            assert a.dtype == b.dtype, f"Retrieved array ({idx}),  wrong dtype!"
            assert np.array_equal(a, b), f"Retrieved array ({idx}), different elements!"


class TestDTensorWithEncodingUtils(object):

    @pytest.fixture()
    def testing_arrays(self):
        return [
            {'size': 0, 'shape': (256, 256, 3)},
            {'size': 0, 'shape': (1000, 1000, 3)},
            {'size': 0, 'shape': (100, 100)},
            {'size': 0, 'shape': (100, 100)},
        ]

    def test_numpy_compressed_images_conversion_into_bundle(self, testing_arrays):

        samples = []
        for test_array in testing_arrays:
            for codec, _ in DTensorUtils.IMAGES_TYPES_MAPPING.items():
                x = np.array(np.random.uniform(0, 255, test_array['shape']), dtype=np.uint8)
                samples.append({'data': x, 'codec': codec})

        images = []
        codecs = []
        for sample in samples:
            images.append(sample['data'])
            codecs.append(sample['codec'])

        action_command = "process_bundle"
        bundle = DTensorUtils.images_to_dtensor_bundle(images, codecs, action_command)
        retrieved_arrays, retrieved_command = DTensorUtils.dtensor_bundle_to_numpy(bundle)

        assert action_command == retrieved_command, f"Wrong retrieved command: '{retrieved_command}'"
        assert len(retrieved_arrays) == len(images), f"Number of retrieved images is wrong! {len(retrieved_arrays)}/{len(images)}"
        for idx in range(len(retrieved_arrays)):
            print("Comparing DTensors arrays:", images[idx].shape, retrieved_arrays[idx].shape)
            a, b = images[idx], retrieved_arrays[idx]
            assert a.shape == b.shape, f"Retrieved array ({idx}),  wrong shape!"
            assert a.dtype == b.dtype, f"Retrieved array ({idx}),  wrong dtype!"
            codec = codecs[idx]
            if not DataCoding.is_codec_lossy(codec):
                assert np.array_equal(a, b), "If codec is lossless, arrays should be equal!"
