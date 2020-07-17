
from persefone.data.databases.mongo.clients import MongoDatabaseClient
from persefone.data.databases.mongo.snapshots import MongoSnapshot, MongoSnapshotCFG
from persefone.data.io.drivers.safefs import SafeFilesystemDriver
import click
import cv2


@click.command("Converts H5 dataset into MongoDB Dataset")
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--driver_cfg', default='database_driver.yml', help="Database IO Driver configuration file", show_default=True)
@click.option('--snapshot_cfg', default='mongo_snapshot_cfg.yml', help="Mongo Snapshot configuration file", show_default=True)
@click.option('--snapshot_key', default='', help="Snapshot key to analyze. Leave blank to show available keys", show_default=True)
def mongo_snapshot_viewer(database_cfg, driver_cfg, snapshot_cfg, snapshot_key):

    driver = SafeFilesystemDriver.create_from_configuration_file(driver_cfg)
    mongo_client = MongoDatabaseClient.create_from_configuration_file(filename=database_cfg)

    snapshot = MongoSnapshot(mongo_client=mongo_client, drivers=[driver], cfg=MongoSnapshotCFG(filename=snapshot_cfg))

    if len(snapshot_key) == 0:
        print("Available keys:", list(snapshot.output_data.keys()))
        return

    assert snapshot_key in snapshot.output_data, f"Key [{snapshot_key}] not found!"

    samples_iterator = snapshot.output_data[snapshot_key]

    # Iterate retrieved samples
    for sample in samples_iterator:

        print("SAMPLE", sample.keys())
        if 'label' in sample:
            print(sample['label'])
        image = sample['x']
        cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        c = cv2.waitKey(0)
        if c == ord('q'):
            break


if __name__ == "__main__":
    mongo_snapshot_viewer()
