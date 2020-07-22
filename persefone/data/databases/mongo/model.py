from mongoengine.fields import (
    Document, StringField,
    DictField, ListField,
    IntField, ReferenceField, DateTimeField
)
from enum import IntEnum


class MDatasetCategory(Document):
    """ Dataset Category Model """

    name = StringField(required=True, unique=True)
    description = StringField()

    def __str__(self) -> str:
        return f"DatasetCategory[{self.name}]"


class MDataset(Document):
    """ Dataset Model """

    name = StringField(required=True, unique=True)
    category = ReferenceField(MDatasetCategory)

    def __str__(self) -> str:
        return f"MDataset[{self.name=},{self.category=}]"


class MResource(Document):
    """ Resource Model. Represents a resource endpoint for a MSample """

    name = StringField()
    driver = StringField(required=True)
    uri = StringField(required=True)


class MSample(Document):
    """ Sample Model holding generic Dataset sample metadata and identifiers """

    sample_id = IntField(required=True, unique_with='dataset')
    metadata = DictField()
    dataset = ReferenceField(MDataset)

    def __str__(self):
        s = f'Sample[{self.dataset.name}_{self.sample_id}]\n'
        for k, v in self.metadata.items():
            s += f'\t {k}\t{v}\n'
        s += '-' * 10
        return s + '\n'


class MItem(Document):
    """ ITem model representing each binary data associated with a single sample """

    sample = ReferenceField(MSample)
    name = StringField(required=True, unique_with='sample')
    resources = ListField(ReferenceField(MResource), ordering='name')


class MTaskStatus(IntEnum):
    """ Available Task statuses """

    READY = 0,
    STARTED = 1,
    WORKING = 2,
    DONE = 3,
    CANCELED = 4,
    UNKNOWN = 1000


class MTask(Document):
    """ Task model representing a queue task with multiple producers/consumers access model """

    source = StringField(required=True)
    name = StringField(required=True, unique=True)
    status = StringField(default=MTaskStatus.UNKNOWN.name)
    created_on = DateTimeField()
    start_time = DateTimeField()
    end_time = DateTimeField()
    priority = IntField(default=0)
    input_payload = DictField(default={})
    working_payload = DictField(default={})
    output_payload = DictField(default={})
    datasets = ListField(ReferenceField(MDataset))


class MModelCategory(Document):
    """ Model Category model """

    name = StringField(required=True, unique=True)
    description = StringField()

    def __str__(self) -> str:
        return f"ModelCategory[{self.name}]"


class MModel(Document):
    """ Model model """

    name = StringField(required=True, unique=True)
    task = ReferenceField(MTask)
    category = ReferenceField(MModelCategory)
    resources = ListField(ReferenceField(MResource), ordering='name')

    def __str__(self) -> str:
        return f"MModel[{self.name=},{self.category=}]"
