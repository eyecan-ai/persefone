
from persefone.data.io.drivers.safefs import SafeFilesystemDriver, SafeFilesystemDriverCFG
import pytest
import numpy as np
from itertools import product
from PIL import Image
import pickle


class TestSafeFS(object):

    def test_file_management(self, safefs_sample_configuration):

        cfg = SafeFilesystemDriverCFG(filename=safefs_sample_configuration)
        driver = SafeFilesystemDriver(cfg)
        print("Driver name", driver.driver_name())

        realms = ['kingdom', 'hell']
        buckets = ['sadopaskdsapod', 'sdaasdsad', 'dsadasdas']
        objects = ['obj0', 'obj1', 'obj2']
        filenames = ['00000.png', '00000.txt', '00000.pt']

        uris = []
        for realm, bucket, objects, filename in product(realms, buckets, objects, filenames):
            uri = driver.uri_from_chunks(realm, bucket, objects, filename)
            uris.append(uri)
            with driver.get(uri, 'wb') as f:
                if 'png' in filename:
                    image = np.random.uniform(0, 255, (256, 256, 3)).astype(np.uint8)
                    image = Image.fromarray(image)
                    image.save(f)
                if 'txt' in filename:
                    data = np.random.uniform(0, 255, (16, 16)).astype(np.uint8)
                    np.savetxt(f, data)
                if 'pt' in filename:
                    data = np.random.uniform(0, 255, (16, 16)).astype(np.uint8)
                    pickle.dump(data, f)

        for uri in uris:
            print(uri)
            driver.delete(uri)
