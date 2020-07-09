
from persefone.interfaces.proto.utils.dtensor import DTensorUtils
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
