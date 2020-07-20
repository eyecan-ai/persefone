from persefone.interfaces.proto.data_pb2 import DTensor, DShape, DType, DTensorBundle
import persefone.interfaces.proto.data_pb2 as proto_data
from persefone.utils.bytes import DataCoding
import numpy as np
from typing import List, Tuple, Union
from numpy import ndarray
import logging


class DTensorUtils(object):

    NUMPY_TYPES_MAPPING = {
        np.dtype(np.uint8).name: proto_data.DT_UINT8,
        np.dtype(np.uint16).name: proto_data.DT_UINT16,
        np.dtype(np.uint32).name: proto_data.DT_UINT32,
        np.dtype(np.int32).name: proto_data.DT_INT32,
        np.dtype(np.int64).name: proto_data.DT_INT64,
        np.dtype(np.float32).name: proto_data.DT_FLOAT,
        np.dtype(np.float64).name: proto_data.DT_DOUBLE,
    }

    NUMPY_TYPES_MAPPING_INV = None

    # Images Types
    IMAGES_TYPES_MAPPING = {
        'jpg': proto_data.IT_JPEG,
        'png': proto_data.IT_PNG,
        'tiff': proto_data.IT_TIFF,
        'bmp': proto_data.IT_BMP,
    }

    IMAGES_TYPES_MAPPING_INV = None

    def __init__(self):
        pass

    @classmethod
    def numpytype_to_dtype(cls, dtype: np.dtype) -> DType:
        """ Converts a numpy dtype to a protobuf DType

        :param dtype: numpy dtype object
        :type dtype: np.dtype
        :raises NotImplementedError: Raises exeception for not mapped dtypes
        :return: DType protobuf object
        :rtype: DType
        """

        sdtype = str(dtype)
        if sdtype in cls.NUMPY_TYPES_MAPPING:
            return cls.NUMPY_TYPES_MAPPING[sdtype]
        else:
            raise NotImplementedError(f'Dtype={sdtype} not supported yet!')

    @classmethod
    def dtype_to_numpytype(cls, dtype: DType) -> np.dtype:
        """ Converst a protobuf DType to numpy dtype

        :param dtype: protobuf DType object to convert
        :type dtype: DType
        :raises NotImplementedError: Raises exception for not mapped dtypes
        :return: corresponding numpy dtype
        :rtype: np.dtype
        """

        if cls.NUMPY_TYPES_MAPPING_INV is None:
            cls.NUMPY_TYPES_MAPPING_INV = {v: k for k, v in cls.NUMPY_TYPES_MAPPING.items()}

        if dtype in cls.NUMPY_TYPES_MAPPING_INV:
            return np.dtype(cls.NUMPY_TYPES_MAPPING_INV[dtype])
        else:
            raise NotImplementedError(f'DTensor Dtype={dtype} not supported yet!')

    @classmethod
    def imagecodec_to_dtype(cls, codec: str) -> DType:
        """ Converts an Image codec, defined by string, in a DType value

        :param codec: source codec (e.g. 'jpeg')
        :type codec: str
        :raises NotImplementedError:  Raises exception for not mapped codecs
        :return: corresponding dtype
        :rtype: DType
        """

        codec = codec.lower()
        if codec in cls.IMAGES_TYPES_MAPPING:
            return cls.IMAGES_TYPES_MAPPING[codec]
        else:
            raise NotImplementedError(f'Codec={codec} not supported yer')

    @classmethod
    def dtype_to_imagecodec(cls, dtype: DType) -> str:
        """ Covnerts a protobuf DType to corresponding image codec

        :param dtype: protobuf DTYpe object to convert
        :type dtype: DType
        :raises NotImplementedError: Raises exception for not mapped dtypes
        :return: corresponding image codec
        :rtype: str
        """

        if cls.IMAGES_TYPES_MAPPING_INV is None:
            cls.IMAGES_TYPES_MAPPING_INV = {v: k for k, v in cls.IMAGES_TYPES_MAPPING.items()}

        if dtype in cls.IMAGES_TYPES_MAPPING_INV:
            return cls.IMAGES_TYPES_MAPPING_INV[dtype]
        else:
            raise NotImplementedError(f'DTensor Dtype={dtype} not supported yet!')

    @classmethod
    def numpyshape_to_dshape(cls, shape: tuple) -> DShape:
        """ Converts numpy shape tuple to DShape object

        :param shape: source numpy shape tuple
        :type shape: tuple
        :return: DShape corresponding object
        :rtype: DShape
        """

        dims = []
        for s in shape:
            dims.append(DShape.Dimension(size=s))
        return DShape(dim=dims)

    @classmethod
    def numpy_to_dtensor(cls, array: ndarray, codec: str = None) -> DTensor:
        """ Converts generic numpy multidimensional array to protobuf DTensor

        :param array: source numpy array
        :type array: ndarray
        :param codec: code used for data
        :type codec: str
        :return: corresponding DTensor object
        :rtype: DTensor
        """

        dtensor = DTensor()
        if codec is None:
            dtensor.dtype = cls.numpytype_to_dtype(array.dtype)
            dshape = cls.numpyshape_to_dshape(array.shape)
            dtensor.shape.CopyFrom(dshape)
            dtensor.content = array.tobytes()
        else:
            dtensor.dtype = cls.imagecodec_to_dtype(codec)
            dshape = cls.numpyshape_to_dshape(array.shape)
            dtensor.shape.CopyFrom(dshape)
            dtensor.content = DataCoding.numpy_image_to_bytes(array, data_encoding=codec)
        return dtensor

    @classmethod
    def dtensor_to_numpy(cls, dtensor: DTensor) -> ndarray:
        """ Converts protobuf DTensor object to numpy array

        :param dtensor: source DTensor
        :type dtensor: DTensor
        :return: corresponding numpy array
        :rtype: ndarray
        """

        shape = dtensor.shape
        dims = []
        for d in shape.dim:
            dims.append(d.size)
        dims = tuple(dims)

        try:
            dtype = cls.dtype_to_numpytype(dtensor.dtype)
            array = np.frombuffer(dtensor.content, dtype).reshape(dims)
            return array
        except NotImplementedError:
            codec = cls.dtype_to_imagecodec(dtensor.dtype)
            array = DataCoding.bytes_to_data(dtensor.content, codec)
            return array

    @classmethod
    def is_valid_dtensor(cls, dtensor: DTensor) -> bool:
        """ Checks is valid DTensor

        :param dtensor: source DTensor
        :type dtensor: DTensor
        :return: TRUE if valid
        :rtype: bool
        """

        try:
            cls.dtensor_to_numpy(dtensor)
            return True
        except Exception as e:
            logging.error(e)
            return False

    @classmethod
    def numpy_to_dtensor_bundle(cls, arrays: List[ndarray], action: str) -> DTensorBundle:
        """ Converst a list of numpy array, with an action string, to protobuf DTensorBundle

        :param arrays: source numpy arrays list
        :type arrays: List[ndarray]
        :param action: generic action string
        :type action: str
        :return: corresponding protobuf DTensorBundle
        :rtype: DTensorBundle
        """

        dtensors = []
        for array in arrays:
            dtensors.append(cls.numpy_to_dtensor(array))

        bundle = DTensorBundle()
        bundle.tensors.extend(dtensors)
        bundle.action = action
        return bundle

    @classmethod
    def images_to_dtensor_bundle(cls, arrays: List[ndarray], codecs: Union[str, List[str]], action: str) -> DTensorBundle:
        """ Converst a list of numpy array (storing images), with relative codecs, with an action string, to protobuf DTensorBundle

        :param arrays: source numpy arrays list
        :type arrays: List[ndarray]
        :param codecs: codec/codecs for images
        :type codecs: Union[str, List[str]]
        :param action: generic action string
        :type action: str
        :return: corresponding protobuf DTensorBundle
        :rtype: DTensorBundle
        """

        if isinstance(codecs, str):
            codecs = [codecs] * len(arrays)

        dtensors = []
        for array, codec in zip(arrays, codecs):
            dtensors.append(cls.numpy_to_dtensor(array, codec=codec))

        bundle = DTensorBundle()
        bundle.tensors.extend(dtensors)
        bundle.action = action
        return bundle

    @classmethod
    def dtensor_bundle_to_numpy(cls, bundle: DTensorBundle) -> Tuple[List[ndarray], str]:
        """ Converts protobuf DTensorBundle to a list of numpy arrays with a generic action string

        :param bundle: source protobuf DTensorBundle
        :type bundle: DTensorBundle
        :return: tuple with (list of numpy arrays, action string)
        :rtype: Tuple[List[ndarray], str]
        """

        arrays = []
        for dtensor in bundle.tensors:
            arrays.append(cls.dtensor_to_numpy(dtensor))

        return arrays, bundle.action
