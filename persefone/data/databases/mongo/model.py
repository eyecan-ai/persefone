from mongoengine.fields import EmbeddedDocument, Document, StringField, DictField, ListField, SortedListField, IntField, ReferenceField


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
    resources = ListField(ReferenceField(MResource), ordering='driver')
