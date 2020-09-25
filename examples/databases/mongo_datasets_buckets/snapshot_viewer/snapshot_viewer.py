
from persefone.data.databases.mongo.nodes.buckets.datasets import DatasetsBucket, DatasetsBucketSnapshot, DatasetsBucketSnapshotCFG
from persefone.data.databases.mongo.clients import MongoDatabaseClientCFG
import click
import cv2


@click.command("Converts H5 dataset into MongoDB Dataset")
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--snapshot_cfg', default='snapshot_mixed_datasets.yml', help="Mongo Snapshot configuration file", show_default=True)
@click.option('--snapshot_key', default='', help="Snapshot key to analyze. Leave blank to show available keys", show_default=True)
def mongo_snapshot_viewer(database_cfg, snapshot_cfg, snapshot_key):

    bucket = DatasetsBucket(client_cfg=MongoDatabaseClientCFG(filename=database_cfg))

    snapshot_cfg = DatasetsBucketSnapshotCFG(filename=snapshot_cfg)
    snapshot = DatasetsBucketSnapshot(bucket, snapshot_cfg)

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
