from mongoengine.errors import NotUniqueError, DoesNotExist
from mongoengine.queryset.queryset import QuerySet
from persefone.data.databases.mongo.model import MDataset, MDatasetCategory, MSample, MItem, MResource
from typing import Union
import logging
from typing import List, Union
import math


class DatasetCategoryRepository(object):

    @classmethod
    def new_category(cls, name, description=''):
        """Creates DatasetCategory

        :param name: category name
        :type name: str
        :param description: category short description, defaults to ''
        :type description: str, optional
        :return: MDatasetCategory instance
        :rtype: MDatasetCategory
        """
        category = MDatasetCategory(name=name, description=description)
        try:
            category.save()
        except NotUniqueError as e:
            logging.error(e)
            category = MDatasetCategory.objects.get(name=name)
        return category

    @classmethod
    def get_category(cls, name, create_if_none=True):
        """Retrieves a MDatasetCategory object from database

        :param name: category name
        :type name: str
        :param create_if_none: set TRUE to create category if not present, defaults to True
        :type create_if_none: bool, optional
        :return: MDatasetCategory instance
        :rtype: MDatasetCategory
        """

        category = None
        try:
            category = MDatasetCategory.objects.get(name=name)
        except DoesNotExist as e:
            if create_if_none:
                category = cls.new_category(name, description='')
            else:
                logging.error(e)
        return category


class DatasetsRepository(object):

    @classmethod
    def new_dataset(cls,
                    dataset_name: str,
                    dataset_category: Union[str, MDatasetCategory]) -> MDataset:
        """ Creates new MDataset

        :param dataset_name: dataset name
        :type dataset_name: str
        :param dataset_category: category name or MDatasetCategory instance (needs a valid ObjectID stored inside)
        :type dataset_category: Union[str, MDatasetCategory]
        :return: MDataset instance
        :rtype: MDataset
        """

        assert dataset_category is not None, "Dataset Category cannot be None!"
        if isinstance(dataset_category, str):
            dataset_category = DatasetCategoryRepository.get_category(name=dataset_category)

        try:
            dataset = MDataset(name=dataset_name, category=dataset_category)
            dataset.save()
        except NotUniqueError as e:
            logging.error(e)
            dataset = cls.get_dataset(dataset_name=dataset_name)

        return dataset

    @classmethod
    def get_dataset(cls, dataset_name: str) -> MDataset:
        """Retrieves MDataset by name

        :param dataset_name: dataset name
        :type dataset_name: str
        :return: MDataset instance
        :rtype: MDataset
        """

        dataset = None
        try:
            dataset = MDataset.objects.get(name=dataset_name)
        except DoesNotExist as e:
            logging.error(e)
        return dataset


class SamplesRepository(object):

    @classmethod
    def new_sample(cls, dataset: MDataset, sample_id: int = -1, metadata: dict = {}) -> Union[MSample, None]:
        """ Creates a MSample associated with target MDataset. The field `sample_id`
        has to be unique among same dataset instances.

        :param dataset: MDateset instance
        :type dataset: MDataset
        :param sample_id: progressive integer, if set to -1 will be auto-generated, defaults to -1
        :type sample_id: int, optional
        :param metadata: sample metadata, defaults to {}
        :type metadata: dict, optional
        :return: MSample instance or None if not unique id
        :rtype: Union[MSample, None]
        """
        if sample_id < 0:
            sample_id = MSample.objects.count()

        sample = MSample(
            sample_id=sample_id,
            metadata=metadata,
            dataset=dataset
        )
        try:
            sample.save()
        except NotUniqueError as e:
            sample = None
            logging.error(e)
        return sample

    @classmethod
    def get_samples(cls, dataset: MDataset) -> QuerySet:
        """ Retrieves list of MSample s of given MDataset

        :param dataset: target MDataset
        :type dataset: MDataset
        :return: QuerySet of associated MSample
        :rtype: QuerySet
        """

        return MSample.objects(dataset=dataset)


class ItemsRepository(object):

    @classmethod
    def new_item(cls, sample: MSample, name: str) -> Union[MItem, None]:
        """ Creates new MItem associated with target MSample

        :param sample: target MSample
        :type sample: MSample
        :param name: unique name among other MSample items
        :type name: str
        :return: created MItem or None if error occurs
        :rtype: Union[MItem, None]
        """

        item = MItem(sample=sample, name=name)
        try:
            item.save()
        except NotUniqueError as e:
            item = None
            logging.error(e)
        return item

    @classmethod
    def get_items(cls, sample: MSample) -> QuerySet:
        """ Retrieves list of MItem s given a target MSample

        :param sample: target MSample
        :type sample: MSample
        :return: QuerySet of associated MItem s
        :rtype: QuerySet
        """
        return MItem.objects(sample=sample).order_by('+name')

    @classmethod
    def create_item_resource(cls, item: MItem, driver: str, uri: str) -> MResource:
        """ Create a MResource associated with target MItem

        :param item: target MItem
        :type item: MItem
        :param driver: resource driver
        :type driver: str
        :param uri: resource uri
        :type uri: str
        :return: created MResource
        :rtype: MResource
        """

        resource = ResourcesRepository._new_resource(driver=driver, uri=uri)
        item.resources.append(resource)
        item.save()


class ResourcesRepository(object):

    @classmethod
    def _new_resource(cls, driver: str, uri: str) -> MResource:
        """ Creates a generic MResource. It is private outside Repositories realm.
        Only another Repository should create a MResource in order to avoid creation
        of orphans resoruces.

        :param driver: driver name
        :type driver: str
        :param uri: uri
        :type uri: str
        :return: created MResource
        :rtype: MResource
        """

        resource = MResource(driver=driver, uri=uri)
        resource.save()
        return resource


class RepositoryTestingData(object):
    DATASET_PREFIX = '##_3213213125r2313_DATASET_'
    CATEGORY_PREFIX = '##_3213213125r2313_CATEGORY_'
    ITEM_PREFIX = '##_3213213125r2313_item_'
    DRIVER_PREFIX = '##_aaadsdsadbasdb123_filesystem_driver_'

    @classmethod
    def dataset_name(cls, dataset_idx):
        return f'{cls.DATASET_PREFIX}{dataset_idx}'

    @classmethod
    def category_name(cls, dataset_idx, n_categories):
        return f'{cls.CATEGORY_PREFIX}{dataset_idx % n_categories}'

    @classmethod
    def item_name(cls, item_idx):
        return f"{cls.ITEM_PREFIX}{item_idx}"

    @classmethod
    def driver_name(cls, resource_idx):
        return f'{cls.DRIVER_PREFIX}{resource_idx}'

    @classmethod
    def metadata(cls, sample_idx):
        return {
            'even': sample_idx % 2,
            'odd': sample_idx % 1,
            'square': sample_idx**2,
            'sqrt': math.sqrt(sample_idx),
            'fake_name': str(sample_idx).zfill(10),
            'seq': [sample_idx, sample_idx + 1, sample_idx * 2, sample_idx + 4],
            'd': {'a': sample_idx**2, 'b': sample_idx / 3}
        }

    @classmethod
    def uri(cls, item_idx, resource_idx):
        if resource_idx % 2 == 0:
            return f'file://a/b/c/{item_idx}/{resource_idx}'
        else:
            return f'amazons3://bucket/bacbdabb3434babcab/{item_idx}/{resource_idx}'

    @classmethod
    def generate_test_data(cls, n_categories: int, n_datasets: int, n_samples: int, n_items: int, n_resources: int):

        for dataset_idx in range(n_datasets):

            category_name = cls.category_name(dataset_idx, n_categories)
            dataset_name = cls.dataset_name(dataset_idx)
            dataset = DatasetsRepository.new_dataset(dataset_name=dataset_name, dataset_category=category_name)

            for sample_idx in range(n_samples):

                metadata = cls.metadata(sample_idx)

                sample = SamplesRepository.new_sample(dataset, sample_idx, metadata)

                for item_idx in range(n_items):

                    item = ItemsRepository.new_item(sample, cls.item_name(item_idx))

                    for resource_idx in range(n_resources):

                        ItemsRepository.create_item_resource(
                            item,
                            cls.driver_name(resource_idx),
                            cls.uri(item_idx, resource_idx)
                        )
