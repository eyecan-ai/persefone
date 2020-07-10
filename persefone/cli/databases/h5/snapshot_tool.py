from persefone.data.databases.snapshot import DatabaseSnapshot, SnapshotConfiguration
from persefone.data.databases.h5 import H5Database, H5SimpleDatabase
from persefone.data.databases.pandas import PandasDatabaseIO
from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDatabaseClientCFG, MongoDataset
from persefone.data.io.drivers.safefs import SafeFilesystemDriver, SafeFilesystemDriverCFG
import json
import yaml
import numpy as np
# cfg = SnapshotConfiguration(filename='/home/daniele/work/workspace_python/persefone/persefone/cli/databases/h5/sample_h5_configuration.yml')
# snapshot = DatabaseSnapshot(cfg=cfg)

# print(snapshot.output_readers)
# reader = snapshot.output_readers['full']
# for item in reader:
#     print(item)
#PandasDatabaseIO.save_csv(snapshot.database, '/tmp/data.csv')
# import time
# import numpy as np

# for i in range(1000):

#     db = H5Database('/tmp/gino.h5', readonly=False)
#     with db:
#         data = np.random.uniform(0, 255, (256, 256, 3))
#         db.store_object(str(i), 'image', data)

#     print("Wrtie")
#     time.sleep(0)

driver = SafeFilesystemDriver.create_from_configuration_file('/home/daniele/work/workspace_python/persefone/tests/sample_data/configurations/drivers/securefs.yml')
mongo_client = MongoDatabaseClient.create_from_configuration_file(filename='/home/daniele/work/workspace_python/persefone/persefone/cli/databases/h5/database.yml')
mongo_dataset = MongoDataset(mongo_client, 'temp_datast', 'cmp', {driver.driver_name(): driver})


db = H5SimpleDatabase(filename='/media/daniele/Data/datasets/cmp/cmp_batch2_h5/green.h5')

with db:
    for h5_sample in db:

        metadata = {}  # TODO: Conversion with external mapper!!!
        for k, v in dict(h5_sample.attrs).items():
            if isinstance(v, np.int64):
                v = int(v)
            metadata[k] = v

        mongo_sample = mongo_dataset.add_sample(metadata=metadata)

        print("Sample: ", mongo_sample.sample_id)

        # print(dict(item.attrs))
        for h5_item_name, h5_data in h5_sample.items():
            mongo_item = mongo_dataset.add_item(mongo_sample.sample_id, h5_item_name)
            mongo_dataset.push_resource_from_blob(
                mongo_sample.sample_id,
                h5_item_name,
                h5_item_name,
                h5_data[...],
                h5_data.attrs['_encoding'],
                driver.driver_name()
            )

            #print(db.load_encoded_data(sample.name, item_name))
