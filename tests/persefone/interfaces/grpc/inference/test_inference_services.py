
import pytest
from persefone.data.databases.mongo.clients import MongoModelsManager, MongoDatabaseTaskManager
from persefone.data.io.drivers.safefs import SafeFilesystemDriver, SafeFilesystemDriverCFG
from persefone.interfaces.grpc.servers.datasets_services import DatasetsServiceCFG
from persefone.interfaces.grpc.servers.inference_services import MongoInferenceService
from persefone.interfaces.grpc.clients.inference_services import InferenceSimpleServiceClient
import grpc

from concurrent import futures
import threading
import numpy as np
import json


class TestMongoInferenceServices(object):

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_lifecycle(self, temp_mongo_database, driver_temp_base_folder, minimnist_folder):
        self._test_lifecycle(temp_mongo_database, driver_temp_base_folder, minimnist_folder)

    @pytest.mark.mongo_mock_server
    def test_lifecycle_mock(self, temp_mongo_mock_database, driver_temp_base_folder, minimnist_folder):
        self._test_lifecycle(temp_mongo_mock_database, driver_temp_base_folder, minimnist_folder)

    def _test_lifecycle(self, mongo_client, driver_temp_base_folder, minimnist_folder):

        host = 'localhost'
        port = 10005

        cfg = SafeFilesystemDriverCFG.from_dict({'base_folder': driver_temp_base_folder})
        driver = SafeFilesystemDriver(cfg)

        cfg = DatasetsServiceCFG()
        service = MongoInferenceService(mongo_client=mongo_client, driver=driver)

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=cfg.options)
        server.add_insecure_port(f'{host}:{port}')

        service.register(server)

        def _serve():
            server.start()
            server.wait_for_termination()

        t = threading.Thread(target=_serve, daemon=True)
        t.start()

        client = InferenceSimpleServiceClient(host, port)

        models = client.models_list()
        assert len(models) == 0, "Models should be emtpy"

        models_manager = MongoModelsManager(mongo_client=mongo_client)
        tasks_manager = MongoDatabaseTaskManager(mongo_client=mongo_client)

        n_models = 10
        for i in range(n_models):
            task = tasks_manager.new_task(f'INF_TASK_{i}')
            assert task is not None, "Task should be valid"
            model = models_manager.new_model(f'INF_MODEL_{i}', f'INF_CAT_{i%2}', task.name)
            assert model is not None, "Model should be valid"

        models = client.models_list()
        assert len(models) == n_models, "Models should be not emtpy"

        # Activations
        for model_name in models:

            with pytest.raises(SystemError):
                client.activate_model(model_name)

            def test_cb(model_name):
                return True

            service.set_lifecycle_callback(test_cb)
            client.activate_model(model_name)

            with pytest.raises(SystemError):
                client.activate_model(model_name + "_IMPOSSIBLE_NAME@@!!")

            service.set_lifecycle_callback(None)

        # DeActivations
        for model_name in models:

            with pytest.raises(SystemError):
                client.deactivate_model(model_name)

            def test_cb(model_name):
                return True

            service.set_lifecycle_callback(test_cb)
            client.deactivate_model(model_name)

            with pytest.raises(SystemError):
                client.deactivate_model(model_name + "_IMPOSSIBLE_NAME@@!!")

            service.set_lifecycle_callback(None)

        # Inference
        def activate_cb(model_name):
            return True

        service.set_lifecycle_callback(activate_cb)

        client.activate_model(models[0])

        n_arrays = 5
        arrays = []
        for arr_idx in range(n_arrays):
            arrays.append(np.random.uniform(0, 255, (100, 100, 3)))
        action = json.dumps({'n_arrays': n_arrays})

        with pytest.raises(SystemError):
            client.inference(arrays, action)

        def inference_cb(arrays, action):
            return arrays, action

        service.set_inference_callback(inference_cb)

        reply_arrays, reply_action = client.inference(arrays, action)

        assert len(reply_arrays) == len(arrays), "Simple bypass should not modify arrays"
        assert reply_action == action, "Simple bypass should not modify action"

        metadata = {'n_arrays': n_arrays}
        reply_arrays, reply_metadata = client.inference_with_metadata(arrays, metadata)
        assert len(reply_arrays) == len(arrays), "Simple bypass should not modify arrays"
        assert reply_metadata == metadata, "Simple bypass should not modify metadata"

        client.deactivate_model(models[0])

        # Service teardown
        server.stop(grace=None)
        t.join()
