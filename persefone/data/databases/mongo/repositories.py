
from mongoengine.errors import NotUniqueError, DoesNotExist
from mongoengine.queryset.queryset import QuerySet
from persefone.data.databases.mongo.model import (
    MDataset, MDatasetCategory, MSample,
    MItem, MResource, MTask, MTaskStatus,
    MModel, MModelCategory
)
import logging
from typing import Union, List
import math
import datetime


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
            logging.info(e)
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

    @classmethod
    def delete_category(cls, name) -> bool:
        """ Deletes target MCategory

        :param name: category name
        :type name: str
        :return: TRUE if deletion is ok
        :rtype: bool
        """

        category = cls.get_category(name, create_if_none=False)
        if category is not None:
            category.delete()
            return True
        return False


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
            logging.info(e)
            dataset = None  # cls.get_dataset(dataset_name=dataset_name)

        return dataset

    @classmethod
    def get_dataset(cls, dataset_name: str) -> Union[MDataset, None]:
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
            logging.error(f'{e} : {dataset_name}')
            dataset = None
        return dataset

    @classmethod
    def get_datasets(cls, dataset_name: str = '') -> List[MDataset]:
        """ Retrieves a list of MDataset with name similar to input dataset_name ()

        :param dataset_name: query string for dataset name [empty for all], defaults to ''
        :type dataset_name: str, optional
        :return: list of MDataset
        :rtype: List[MDataset]
        """

        return MDataset.objects(name__contains=dataset_name)

    @classmethod
    def delete_dataset(cls, dataset_name: str) -> bool:
        """ Deletes target MDataset and all nested documents

        :param dataset_name: target dataset Name
        :type dataset_name: str
        :return: TRUE if deletion is complete
        :rtype: bool
        """

        dataset = cls.get_dataset(dataset_name)
        if dataset is not None:
            dataset: MDataset
            samples = SamplesRepository.get_samples(dataset)
            for sample in samples:
                sample: MSample
                items = ItemsRepository.get_items(sample)
                for item in items:
                    item: MItem
                    for resource in item.resources:
                        resource: MResource
                        resource.delete()
                    item.delete()
                sample.delete()
            dataset.delete()
            return True

        return False


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
            sample_id = SamplesRepository.count_samples(dataset=dataset)

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
    def get_samples(cls, dataset: Union[MDataset, None] = None, query_dict: dict = {}, order_bys: list = []) -> QuerySet:
        """ Retrieves list of MSample s of given MDataset

        :param dataset: target MDataset or None for all
        :type dataset: Union[MDataset, None]
        :return: QuerySet of associated MSample
        :rtype: QuerySet
        """

        if dataset is None:
            return MSample.objects(**query_dict).order_by(*order_bys)
        else:
            return MSample.objects(dataset=dataset, **query_dict).order_by(*order_bys)

    @classmethod
    def count_samples(cls, dataset: Union[MDataset, None] = None) -> int:
        """ Count samples

        :param dataset: target MDataset or none, defaults to None
        :type dataset: Union[MDataset, None], optional
        :return: number of samples
        :rtype: int
        """
        if dataset is None:
            # return len(list(MSample.objects()))
            try:
                return MSample.objects().count()
            except Exception as e:
                logging.error(e)
                return len(list(MSample.objects()))
        else:
            # return len(list(MSample.objects(dataset=dataset)))
            try:
                return MSample.objects(dataset=dataset).count()
            except Exception as e:
                logging.error(e)
                return len(list(MSample.objects(dataset=dataset)))

    @classmethod
    def get_sample_by_idx(cls, dataset: MDataset, idx: int) -> Union[MSample, None]:
        """ Gets single MSample associated to target data

        :param dataset: target dataset
        :type dataset: MDataset
        :param idx: sample id
        :type idx: int
        :return: retrived MSample or None
        :rtype: Union[MSample, None]
        """
        sample = None
        try:
            sample = MSample.objects(dataset=dataset, sample_id=idx).get()
        except DoesNotExist as e:
            logging.error(e)
            sample = None
        return sample

    @classmethod
    def delete_sample(cls, dataset_name: str, sample_idx: int) -> bool:
        """ Delete target MSample and its children

        :param dataset_name: target dataset
        :type dataset_name: str
        :param sample_idx: target sample idx
        :type sample_idx: int
        :return: TRUE if deletion is ok
        :rtype: bool
        """

        dataset = DatasetsRepository.get_dataset(dataset_name)
        if dataset is not None:
            sample: MSample = cls.get_sample_by_idx(dataset, sample_idx)
            if sample is not None:
                items = ItemsRepository.get_items(sample)
                for item in items:
                    item: MItem
                    for resource in item.resources:
                        resource.delete()
                    item.delete()
                sample.delete()
                return True
        return False


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
            logging.info(e)
        return item

    @classmethod
    def get_items(cls, sample: Union[MSample, None] = None) -> QuerySet:
        """ Retrieves list of MItem s given a target MSample

        :param sample: target MSample or None for all Items
        :type sample: Union[MSample, None]
        :return: QuerySet of associated MItem s
        :rtype: QuerySet
        """

        if sample is None:
            return MItem.objects().order_by('+name')
        else:
            return MItem.objects(sample=sample).order_by('+name')

    @classmethod
    def get_item_by_name(cls, sample: MSample, name: str) -> Union[MItem, None]:
        """ Gets single item by name

        :param sample: target sample
        :type sample: MSample
        :param name: item name
        :type name: str
        :return: MItem or None
        :rtype: Union[MItem, None]
        """
        item = None
        try:
            item = MItem.objects(sample=sample, name=name).get()
        except DoesNotExist as e:
            logging.error(e)
            item = None
        return item

    @classmethod
    def create_item_resource(cls, item: MItem, name: str, driver: str, uri: str) -> Union[MResource, None]:
        """ Create a MResource associated with target MItem

        :param item: target MItem
        :type item: MItem
        :param driver: resource driver
        :type driver: str
        :param name: resource name
        :type name: str
        :param uri: resource uri
        :type uri: str
        :return: created MResource, NONE if duplicate resource is present with the same name
        :rtype: MResource
        """

        for r in item.resources:
            if r.name == name:
                return None

        resource = ResourcesRepository._new_resource(name=name, driver=driver, uri=uri)
        item.resources.append(resource)
        item.save()
        return resource


class ResourcesRepository(object):

    @classmethod
    def _new_resource(cls, name: str, driver: str, uri: str) -> MResource:
        """ Creates a generic MResource. It is private outside Repositories realm.
        Only another Repository should create a MResource in order to avoid creation
        of orphans resoruces.

        :param name: resource name
        :type name: str
        :param driver: driver name
        :type driver: str
        :param uri: uri
        :type uri: str
        :return: created MResource
        :rtype: MResource
        """

        resource = MResource(name=name, driver=driver, uri=uri)
        resource.save()
        return resource


class TasksRepository(object):

    @classmethod
    def new_task(cls, name: str, source: str, input_payload: dict = {}) -> MTask:
        """ Creates new MTask object

        :param name: target name
        :type name: str
        :param source: source name
        :type source: str
        :param input_payload: an input paylod to link with task, defaults to {}
        :type input_payload: dict, optional
        :return: stored MTask
        :rtype: MTask
        """

        task = MTask(name=name, source=source)
        task.created_on = datetime.datetime.now()
        task.status = MTaskStatus.READY.name
        task.input_payload = input_payload

        try:
            task.save()
        except NotUniqueError as e:
            logging.info(e)
            task = None

        return task

    @classmethod
    def get_tasks(cls, status: Union[MTaskStatus, None] = None, last_first=True, negate: bool = False) -> QuerySet:
        """ Retrieves tasks

        :param status: retrievs only MTask s with target MTaskStatus, defaults to None
        :type status: Union[MTaskStatus, None], optional
        :param last_first: TRUE to order last task first, defaults to True
        :type last_first: bool, optional
        :param negate: TRUE to negate results, defaults to False
        :type negate: bool, optional
        :return: associated QuerySet
        :rtype: QuerySet
        """

        order_by = f'{"-" if last_first else "+"}created_on'

        if status is None:
            return MTask.objects().order_by(order_by)
        else:
            if isinstance(status, MTaskStatus):
                if not negate:
                    return MTask.objects(status=status.name).order_by(order_by)
                else:
                    return MTask.objects(status__ne=status.name).order_by(order_by)

            if isinstance(status, list):
                if not negate:
                    return MTask.objects(status__in=[s.name for s in status]).order_by(order_by)
                else:
                    return MTask.objects(status__nin=[s.name for s in status]).order_by(order_by)

            return MTask.objects(status=status.name).order_by(order_by)

    @classmethod
    def get_task_by_name(cls, name: str) -> Union[MTask, None]:
        """ Retrieves a MTask by name

        :param name: target name
        :type name: str
        :return: retrieved MTask or None
        :rtype: MTask
        """

        try:
            return MTask.objects(name=name).get()
        except DoesNotExist as e:
            logging.error(e)
            return None

    @classmethod
    def start_task(cls, task: MTask) -> Union[MTask, None]:
        """ Start target MTask setting status to MTaskStatus.STARTED

        :param task: target task
        :type task: MTask
        :return: the input task if starting is ok, NONE if start is not allowed
        :rtype: Union[MTask, None]
        """

        if task.status == MTaskStatus.READY.name:
            task.status = MTaskStatus.STARTED.name
            task.start_time = datetime.datetime.now()
            task.save()
            return task
        else:
            return None

    @classmethod
    def work_on_task(cls, task: MTask, working_payload: dict = {}) -> Union[MTask, None]:
        """ Work on target MTask setting status on MTaskStatus.WORKING

        :param task: target MTask
        :type task: MTask
        :param working_payload: associated working metadata, defaults to {}
        :type working_payload: dict, optional
        :return: the input task if working is ok, NONE if working is not allowed
        :rtype: Union[MTask, None]
        """

        good_statuses = [
            MTaskStatus.STARTED.name,
            MTaskStatus.WORKING.name
        ]
        if task.status in good_statuses:
            task.status = MTaskStatus.WORKING.name
            task.start_time = datetime.datetime.now()
            task.working_payload = working_payload
            task.save()
            return task
        else:
            return None

    @classmethod
    def complete_task(cls, task: MTask, output_payload: dict = {}) -> Union[MTask, None]:
        """ Completes target MTask setting status to MTaskStatus.DONE

        :param task: target MTask
        :type task: MTask
        :param output_payload: complete task associated metadata, defaults to {}
        :type output_payload: dict, optional
        :return: the input task if complete is ok, NONE if complete is not allowed
        :rtype: Union[MTask, None]
        """

        good_statuses = [
            MTaskStatus.STARTED.name,
            MTaskStatus.WORKING.name
        ]
        if task.status in good_statuses:
            task.status = MTaskStatus.DONE.name
            task.end_time = datetime.datetime.now()
            task.output_payload = output_payload
            task.save()
            return task
        else:
            return None

    @classmethod
    def cancel_task(cls, task: MTask) -> Union[MTask, None]:
        """ Cancel target MTask setting status to MTaskStatus.CANCELED

        :param task: target MTask
        :type task: MTask
        :return:  the input task if complete is ok, NONE if cancel is not allowed
        :rtype: Union[MTask, None]
        """

        good_statuses = [
            MTaskStatus.READY.name,
            MTaskStatus.STARTED.name,
            MTaskStatus.WORKING.name
        ]
        if task.status in good_statuses:
            task.status = MTaskStatus.CANCELED.name
            task.end_time = datetime.datetime.now()
            task.save()
            return task
        else:
            return None

    @classmethod
    def delete_task(cls, task: MTask) -> bool:
        """ Deletes target task

        :param task: target task
        :type task: MTask
        :return: TRUE for deletion complete, FALSE otherwise
        :rtype: bool
        """

        try:
            task.delete()
        except Exception as e:
            logging.error(e)
            return False
        return True

    @classmethod
    def get_tasks_by_dataset(cls, datasets: Union[MDataset, List[MDataset]]) -> QuerySet:
        """ Retrieves MTask list by associated MDataset list

        :param datasets: list or single MDataset instance
        :type datasets: Union[MDataset, List[MDataset]]
        :return: MTask list
        :rtype: List[MTask]
        """
        if not isinstance(datasets, list):
            datasets = [datasets]

        tasks = []
        try:
            tasks = MTask.objects(datasets__in=datasets)
        except DoesNotExist as e:
            tasks = []
            logging.error(e)
        return tasks


class ModelCategoryRepository(object):

    @classmethod
    def new_category(cls, name, description='') -> MModelCategory:
        """Creates MModelCategory

        :param name: category name
        :type name: str
        :param description: category short description, defaults to ''
        :type description: str, optional
        :return: MModelCategory instance
        :rtype: MModelCategory
        """

        category = MModelCategory(name=name, description=description)
        try:
            category.save()
        except NotUniqueError as e:
            logging.info(e)
            category = MModelCategory.objects.get(name=name)
        return category

    @classmethod
    def get_category(cls, name, create_if_none=True):
        """Retrieves a MModelCategory object from database

        :param name: category name
        :type name: str
        :param create_if_none: set TRUE to create category if not present, defaults to True
        :type create_if_none: bool, optional
        :return: MModelCategory instance
        :rtype: MModelCategory
        """

        category = None
        try:
            category = MModelCategory.objects.get(name=name)
        except DoesNotExist as e:
            if create_if_none:
                category = cls.new_category(name, description='')
            else:
                logging.error(e)
        return category

    @classmethod
    def get_categories(cls, name: str = '') -> QuerySet:
        """ Retrieves categories by query string

        :param name: query string, defaults to ''
        :type name: str, optional
        :return: Set of MModelCategories
        :rtype: QuerySet
        """

        return MModelCategory.objects(name__contains=name)


class ModelsRepository(object):

    @classmethod
    def new_model(cls, model_name: str, model_category: Union[str, MModelCategory]) -> MModel:
        """ Creates new MModel

        :param model_name: model name
        :type model_name: str
        :param model_category: model category (string value or reference to MModelCategory)
        :type model_category: Union[str, MModelCategory]
        :return: MModel instance
        :rtype: MModel
        """

        assert model_category is not None, "Model Category cannot be None!"

        if isinstance(model_category, str):
            model_category = ModelCategoryRepository.get_category(name=model_category)

        try:
            model = MModel(name=model_name, category=model_category)
            model.save()
        except NotUniqueError as e:
            logging.info(e)
            model = None
        return model

    @classmethod
    def get_model(cls, model_name: str) -> Union[MModel, None]:
        """Retrieves MModel by name

        :param dataset_name: model name
        :type dataset_name: str
        :return: MModel instance
        :rtype: MModel
        """

        model = None
        try:
            model = MModel.objects.get(name=model_name)
        except DoesNotExist as e:
            model = None
            logging.error(e)
        return model

    @classmethod
    def get_models(cls, model_category: MModelCategory = None) -> QuerySet:
        """Retrieves MModel list by category if any

        :param model_category: model category
        :type model_category: str
        :return: MModel list
        :rtype:  List[MModel]
        """

        models = []
        try:
            if model_category is not None:
                models = MModel.objects(category=model_category)
            else:
                models = MModel.objects()
        except DoesNotExist as e:
            models = []
            logging.error(e)
        return models

    @classmethod
    def delete_model(cls, name: str) -> bool:
        """ Delete MModel

        :param name: model name
        :type name: str
        :return: TRUE if deletion occurs
        :rtype: bool
        """

        model = cls.get_model(model_name=name)
        if model is not None:
            try:
                model.delete()
                return True
            except Exception as e:
                logging.error(e)
        return False

    @classmethod
    def get_models_by_dataset(cls, datasets: Union[MDataset, List[MDataset]]) -> QuerySet:
        """ Retrieves MModel list by associated MDataset list

        :param datasets: list or single MDataset instance
        :type datasets: Union[MDataset, List[MDataset]]
        :return: MModel list
        :rtype: List[MModel]
        """
        if not isinstance(datasets, list):
            datasets = [datasets]

        tasks = TasksRepository.get_tasks_by_dataset(datasets=datasets)
        return MModel.objects(task__in=tasks)

    @classmethod
    def get_model_by_task(cls, task: MTask) -> Union[MModel, None]:
        """ Retrieves singel MModel associated with task

        :return: Retrived MModel
        :rtype: Union[MModel, None]
        """

        try:
            return MModel.objects.get(task=task)
        except Exception as e:
            logging.error(e)
            return None

    @classmethod
    def create_model_resource(cls, model: MModel, name: str, driver: str, uri: str) -> Union[MResource, None]:
        """ Creates a MResource associated with target MModel

        :param model: target MModel
        :type model: MModel
        :param name: resource name (should be unique among model resources)
        :type name: str
        :param driver: driver name
        :type driver: str
        :param uri: uri
        :type uri: str
        :return: created MResource instance, NONE if duplicate resource was found
        :rtype: Union[MResource, None]
        """

        for r in model.resources:
            if r.name == name:
                return None

        resource = ResourcesRepository._new_resource(name=name, driver=driver, uri=uri)
        model.resources.append(resource)
        model.save()
        return resource


#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################


class RepositoryTestingData(object):
    DATASET_PREFIX = '##_3213213125r2313_DATASET_'
    TASK_PREFIX = '##_3213213125r2313_TASK_'
    MODEL_PREFIX = '##_3213213125r2313_MODEL_'
    CATEGORY_PREFIX = '##_3213213125r2313_CATEGORY_'
    ITEM_PREFIX = '##_3213213125r2313_item_'
    RESOURCE_PREFIX = '##_resource_name_'
    DRIVER_PREFIX = '##_aaadsdsadbasdb123_filesystem_driver_'

    @classmethod
    def dataset_name(cls, dataset_idx):
        return f'{cls.DATASET_PREFIX}{dataset_idx}'

    @classmethod
    def model_name(cls, model_idx):
        return f'{cls.MODEL_PREFIX}{model_idx}'

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
    def resource_name(cls, resource_idx):
        return f'{cls.RESOURCE_PREFIX}{resource_idx}'

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
    def generate_test_data(cls,
                           n_categories: int,
                           n_datasets: int,
                           n_samples: int,
                           n_items: int,
                           n_resources: int,
                           n_models_categories: int = 4,
                           n_models: int = 5,
                           n_model_resources: int = 2):

        created_datasets = []

        for dataset_idx in range(n_datasets):

            category_name = cls.category_name(dataset_idx, n_categories)
            dataset_name = cls.dataset_name(dataset_idx)
            dataset = DatasetsRepository.new_dataset(dataset_name=dataset_name, dataset_category=category_name)
            assert dataset is not None, "Dataset creation fails!"
            created_datasets.append(dataset)

            for sample_idx in range(n_samples):

                metadata = cls.metadata(sample_idx)

                sample = SamplesRepository.new_sample(dataset, sample_idx, metadata)
                assert sample is not None, "Delete old testing data!!"

                for item_idx in range(n_items):

                    item = ItemsRepository.new_item(sample, cls.item_name(item_idx))

                    for resource_idx in range(n_resources):

                        ItemsRepository.create_item_resource(
                            item,
                            cls.resource_name(resource_idx),
                            cls.driver_name(resource_idx),
                            cls.uri(item_idx, resource_idx)
                        )

        created_tasks = []
        for task_idx in range(n_models):
            task = TasksRepository.new_task(f'{cls.TASK_PREFIX}_{task_idx}', "GenerateTestData@")
            for dataset_idx, dataset in enumerate(created_datasets):
                task.datasets.append(dataset)
            task.save()
            created_tasks.append(task)

        assert len(TasksRepository.get_tasks_by_dataset(created_datasets[0])) == len(created_tasks), "Tasks by dataset is wrong"

        for model_idx in range(n_models):

            model_category_name = cls.category_name(model_idx, n_models_categories)
            model_name = cls.model_name(model_idx)
            model = ModelsRepository.new_model(model_name, model_category_name)
            model.task = created_tasks[model_idx]

            for dataset in created_datasets:
                model.task.datasets.append(dataset)
                model.save()

            for resource_idx in range(n_model_resources):
                ModelsRepository.create_model_resource(
                    model,
                    cls.resource_name(resource_idx),
                    cls.driver_name(resource_idx),
                    cls.uri(model_idx, resource_idx)
                )
