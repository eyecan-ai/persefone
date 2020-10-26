import numpy as np
from PIL import Image
import click
import uuid
import sys
import os
import functools
from typing import Callable, Dict, Optional, Sequence, Tuple
from collections import OrderedDict, defaultdict
from pathlib import Path

import yaml
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QSize, QThread
from PyQt5.QtWidgets import (
    QApplication, QWidget, QListWidget, QLabel, QListWidgetItem, QInputDialog, QMessageBox,
    QTableView, QProgressBar
)
from PyQt5.QtGui import QMovie, QPixmap, QIcon, QStandardItem, QStandardItemModel

import persefone
from persefone.data.databases.mongo.nodes.buckets.datasets import DatasetsBucket
from persefone.data.databases.mongo.clients import MongoDatabaseClientCFG
from persefone.data.databases.mongo.nodes.nodes import MNode
from persefone.utils.bytes import DataCoding


def get_absolute_path(relative_path: str):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, relative_path)


def show_error_message(text: str, info: str = '', title: str = 'Error'):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(text)
    msg.setInformativeText(info)
    msg.setWindowTitle(title)
    msg.exec_()


def flatten(d: dict,
            parent_key: str = '',
            sep: str = '.') -> dict:
    """ Flattens a dictionary so that every value consists of a single value
    :param d: the dictionary to flatten
    :param parent_key: used for recursion, represents the name of the current dict's father
    :param sep: the separator used to separate parent keys from their children
    :return: a flattened dictionary
    """

    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = sep.join((parent_key, str(k))) if parent_key else str(k)
            items.extend(flatten(v, new_key, sep=sep).items())
    else:
        items.extend([(parent_key, d)])
    return dict(items)


class DropLabel(QLabel):
    DROP_SIGNAL = QtCore.pyqtSignal([list])

    def __init__(self, parent: QWidget = None, text: str = "Drop new samples"):
        """ Widget with DROP area for files

        :param parent: parent QWidget, defaults to None
        :type parent: QWidget, optional
        """
        super(DropLabel, self).__init__(parent)

        self.setAcceptDrops(True)
        self.setText(text)
        self.setStyleSheet("border:2px solid #E57373")
        self.setStyleSheet("background: #fafafa")  # TODO: hardcoded colors!
        self.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def dragEnterEvent(self, event):
        self.setStyleSheet("background: #E57373")
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("background: #fafafa")

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        self.setStyleSheet("background: #fafafa")
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
            file_list = []
            for url in event.mimeData().urls():
                file_list.append(str(url.toLocalFile()))
            self.DROP_SIGNAL.emit(file_list)
        else:
            event.ignore()


class EventLoop(object):

    CALLBACKS_MAP = {}
    FALLBACKS_MAP = {}
    _call_thread = None

    @classmethod
    def defer_call(cls, method: Callable, callback: Callable, fallback: Optional[Callable] = None) -> None:
        """ Defers a call by means of QThread

        :param event_name: event name used to track response
        :type event_name: str
        :param method: generic function
        :type method: Callable
        :param callback: callback function
        :type callback: Callable
        :param fallback: optional fallback function, defaults to None
        :type fallback: Optional[Callable]
        """

        uinque_name = str(uuid.uuid1())
        cls.CALLBACKS_MAP[uinque_name] = callback
        cls.FALLBACKS_MAP[uinque_name] = fallback
        if cls._call_thread is None:
            cls._call_thread = CallThread(event_name=uinque_name, method=method)
            cls._call_thread.CALL_END.connect(cls.on_deferred_response)
            cls._call_thread.start()

    @classmethod
    def on_deferred_response(cls, event_name, data, error):
        cls._call_thread = None
        callback = cls.CALLBACKS_MAP[event_name]
        fallback = cls.FALLBACKS_MAP[event_name]
        if error is not None:
            if fallback is not None:
                fallback(error)
            else:
                raise error
        else:
            callback(data)


class CallThread(QThread):
    CALL_END = QtCore.pyqtSignal(str, object, object)

    def __init__(self, event_name, method):
        super(CallThread, self).__init__()
        self._method = method
        self._event_name = event_name

    def run(self):
        """ Deferred call """
        try:
            result = self._method()
        except Exception as e:
            self.CALL_END.emit(self._event_name, None, e)
            return
        self.CALL_END.emit(self._event_name, result, None)


class UploadSamplesThread(QThread):
    PROGRESS_CHANGED = QtCore.pyqtSignal(int)
    PROCESS_END = QtCore.pyqtSignal(bool)

    def __init__(self,
                 datasets_bucket: DatasetsBucket,
                 dataset: MNode,
                 folder: str) -> None:
        """ Creates an Upload Thread to upload new datasets samples

        :param datasets_bucket: the datasets bucket to which upload the new samples
        :type client: DatasetsBucket
        :param dataset: target dataset node
        :type dataset: MNode
        :param samples: list of samples to upload, defaults to None
        :type samples: Sequence[EDatasetSample], optional
        """

        super(UploadSamplesThread, self).__init__()
        self._datasets_bucket = datasets_bucket
        self._dataset = dataset
        self._folder = folder

    def run(self):
        """ Upload Loop """

        tree = DatasetsUtils.tree_from_underscore_notation_files(self._folder)

        i = 0

        for sample_id, sample_data in tree.items():
            sample_id = str(int(sample_id))

            sample_metadata = {}
            for tag, filename in sample_data.items():
                f = Path(filename)
                if 'yml' in f.suffix:
                    with open(f, 'r') as metadata_file:
                        sample_metadata[tag] = yaml.safe_load(metadata_file)

            sample_node = self._datasets_bucket.new_sample(self._dataset.last_name, sample_metadata, sample_id)

            for tag, filename in sample_data.items():
                f = Path(filename)
                for encoding in ['jpg', 'png']:
                    if encoding in f.suffix:
                        image = np.array(Image.open(str(f)))
                        blob = DataCoding.numpy_image_to_bytes(image, encoding)
                        sample_node.metadata[f'item_{tag}_shape'] = list(image.shape)
                        sample_node.metadata[f'item_{tag}_blobsize'] = len(blob)
                        sample_node.metadata[f'item_{tag}_encoding'] = encoding
                        sample_node.save()
                        self._datasets_bucket.new_item(self._dataset.last_name, sample_id, tag, blob, encoding)

            # Compute and emit progress value
            progress = int((i / len(tree) * 100))
            i += 1
            self.PROGRESS_CHANGED.emit(progress)

        # emit progress 100%
        self.PROGRESS_CHANGED.emit(100)

        # emit progress END
        self.PROCESS_END.emit(True)


class DatasetsUtils(object):

    @classmethod
    def tree_from_underscore_notation_files(cls, folder: str, skip_hidden_files=True) -> dict:
        """Walk through files in folder generating a tree based on Underscore notation.
        Leafs of abovementioned tree are string representing filenames, inner nodes represent
        keys hierarchy.

        :param folder: target folder
        :type folder: str
        :param skip_hidden_files: TRUE to skip files starting with '.', defaults to True
        :type skip_hidden_files: bool, optional
        :return: dictionary representing multilevel tree
        :rtype: dict
        """

        # Declare TREE structure
        def tree():
            return defaultdict(tree)

        keys_tree = tree()
        folder = Path(folder)
        files = list(sorted(folder.glob('*')))
        for f in files:
            name = f.stem

            if skip_hidden_files:
                if name.startswith('.'):
                    continue

            chunks = name.split('_', maxsplit=1)
            if len(chunks) == 1:
                chunks.append('none')
            p = keys_tree
            for index, chunk in enumerate(chunks):
                if index < len(chunks) - 1:
                    p = p[chunk]
                else:
                    p[chunk] = str(f)

        return dict(keys_tree)


class DatasetsBrowser(QWidget):

    def __init__(self, datasets_bucket: DatasetsBucket, parent=None):
        super(DatasetsBrowser, self).__init__(parent)
        self.ui = uic.loadUi(get_absolute_path('data/datasets_browser.ui'), self)

        self.setWindowTitle("Persefone - Datasets Browser")
        self.label_logo: QLabel()
        self.show_logo()

        self._sample_id_name_mapping = {}

        # Create Drop Files ZONE
        drop_label = DropLabel(text="Drop folder here!")
        self.drop_zone.addWidget(drop_label)
        drop_label.DROP_SIGNAL.connect(self.on_dropped_folder)

        # Dataset client
        self._datasets_bucket = datasets_bucket

        # Buttons
        self.btn_list.clicked.connect(self.show_dataset_list)
        self.btn_new_dataset.clicked.connect(self.new_dataset)
        self.btn_delete_dataset.clicked.connect(self.delete_current_dataset)
        # self.btn_test.clicked.connect(self.generate_snapshot_file)

        # List widgets
        self.datasets_list: QListWidget
        self.samples_list: QListWidget
        self.items_list: QListWidget
        self.sample_metadata: QTableView

        # Labels
        self.label_title: QLabel
        self.label_title.setText(f'Persefone Datasets Browser v{persefone.__version__}')

        # Upload progress bar
        self.upload_progress: QProgressBar

        # Lists events
        self.datasets_list.currentRowChanged.connect(self.on_dataset_selected)
        self.samples_list.currentRowChanged.connect(self.on_sample_selected)
        self.items_list.currentRowChanged.connect(self.on_item_selected)

        # Dataset
        self._cached_datasets: Dict[str, MNode] = {}
        self._cached_samples: Dict[str, MNode] = {}
        self._cached_items: Dict[str, MNode] = {}
        self._active_dataset: MNode = None
        self._active_sample: MNode = None

        # GRPC Calls
        self._grpc_call_thread = None
        self._upload_thread = None

    def show_logo(self):
        """ Shows logo """

        self.label_logo.setText(f"<img height=32 src='{get_absolute_path('data/icons/logo.png')}'></img>")

    def show_loading(self):
        """ Shows loading """

        movie = QMovie(get_absolute_path('data/icons/loading.gif'))
        movie.setScaledSize(QSize(32, 32))
        self.label_logo.setMovie(movie)
        movie.start()

    def set_loading_status(self, status: bool = True):
        if status:
            self.show_loading()
        else:
            self.show_logo()

    def delete_current_dataset(self):
        """ Deletes active dataset """

        # Dialog for dataset name input
        name, status = QInputDialog.getText(self, "Delete dataset", "Security Check. Write dataset name:")
        if status:  # Dialog OK pressed
            if self._active_dataset is not None:  # is at least one dataset active?
                # security check on name before delete
                if name == self._active_dataset.last_name:
                    self.set_loading_status(True)

                    def _deferred_delete_dataset(*args) -> None:
                        self.clear_active_dataset()
                        self.show_dataset_list()
                        self.set_loading_status(False)

                    EventLoop.defer_call(
                        functools.partial(self._datasets_bucket.delete_dataset_fast, self._active_dataset.last_name),
                        _deferred_delete_dataset,
                        fallback=self.show_error_dialog
                    )

    def show_dataset_list(self):
        """ Fetches dataset list from database """

        # Loading Status ON
        self.set_loading_status(True)

        def _deferred_datasets_list(datasets):
            self.clear_active_dataset()
            self.datasets_list.clear()
            self._cached_datasets = {d.last_name: d for d in datasets}
            # Populates list
            for _, dataset in self._cached_datasets.items():
                self.datasets_list.addItem(QListWidgetItem(QIcon(get_absolute_path("data/icons/dataset.png")), dataset.last_name))

            # Loading Status OFF
            self.set_loading_status(False)

        # Deferred
        EventLoop.defer_call(self._datasets_bucket.get_datasets,
                             _deferred_datasets_list,
                             fallback=self.show_error_dialog)

    def show_error_dialog(self, error: Exception) -> None:
        """Shows a dialog that displays an error message

        :param error: the encountered error
        :type error: Exception
        """
        dialog = QMessageBox()
        dialog.setIcon(QMessageBox.Critical)
        dialog.setText(str(error))
        dialog.setWindowTitle('Error')
        dialog.exec_()

    def on_dataset_selected(self, index):
        """ On dataset selected on list

        :param index: selected index
        :type index: any
        """

        if self.datasets_list.currentItem() is not None:
            dataset_name = self.datasets_list.currentItem().text()
            self.reload_dataset(dataset_name)

    def clear_active_dataset(self):
        """ Clears all views relative to active dataset """

        self._active_dataset = None
        self._active_sample = None
        self.show_samples_list()
        self.show_items_list()
        self.set_sample_image(None)

    def reload_dataset(self, dataset_name: str):
        """ Reload a dataset by name

        :param dataset_name: target dataset name
        :type dataset_name: str
        """

        if len(self._cached_datasets) > 0:
            if dataset_name in self._cached_datasets:
                dataset = self._cached_datasets[dataset_name]
                self.clear_active_dataset()
                self._active_dataset = dataset
                self._active_sample = None
                self.show_samples_list()

    def show_samples_list(self):
        """ Shows active dataset samples list """

        # clear previous list
        self.samples_list.clear()
        self._sample_id_name_mapping = {}

        # Creates new table for metadata
        model = QStandardItemModel(self)
        self.sample_metadata.setModel(model)

        if self._active_dataset is not None:  # is dataset active?

            self.set_loading_status()

            def _deferred_samples_list(samples):

                sample_ids = []
                self._cached_samples = {}

                for sample in samples:
                    sample_id = sample.metadata['#sample_id']
                    self._cached_samples[sample_id] = sample
                    sample_ids.append(sample_id)

                for sample_id in sorted(sample_ids):
                    name = f'Sample_{sample_id}'
                    self.samples_list.addItem(QListWidgetItem(QIcon(), name))
                    self._sample_id_name_mapping[name] = sample_id

                self.set_loading_status(False)

            EventLoop.defer_call(
                functools.partial(self._datasets_bucket.get_samples, self._active_dataset.last_name),
                _deferred_samples_list,
                fallback=self.show_error_dialog
            )

    def update_metadata(self, metadata: dict):
        """ Update metadata table

        :param metadata: metadata to show
        :type metadata: dict
        """

        # Clear metadata table
        model = QStandardItemModel(self)
        self.sample_metadata.setModel(model)
        self.sample_metadata.verticalHeader().setVisible(False)
        self.sample_metadata.horizontalHeader().setVisible(False)

        # flatten/ordered metadata
        metadata = flatten(metadata)
        metadata = OrderedDict(sorted(metadata.items()))

        # populates rows
        for k, v in metadata.items():
            item = QStandardItem(k)
            item_v = QStandardItem(str(v))
            row = []
            row.append(item)
            row.append(item_v)
            model.appendRow(row)

        # repack table
        self.sample_metadata.resizeColumnsToContents()

    def on_sample_selected(self, index):
        """ On sample selected event

        :param index: sample index
        :type index: any
        """

        if self.samples_list.currentItem() is not None:  # Selection is valid?
            if self._active_dataset is not None:  # Is dataset active?

                # Builds id based on item position
                sample_name = self.samples_list.currentItem().text()
                sample_id = self._sample_id_name_mapping[sample_name]

                self.set_loading_status()

                def _deferred_get_sample(sample_with_data: Tuple[MNode, Sequence[MNode]]):
                    sample_node, item_nodes = sample_with_data

                    self._active_sample = sample_node
                    if sample_node is not None:  # Valid sample

                        self.update_metadata(sample_node.metadata)
                        self._cached_items = {item.last_name: item for item in item_nodes}

                        # update items
                        self.show_items_list()

                    self.set_loading_status(False)

                def _get_sample_with_data() -> Tuple[MNode, Sequence[MNode]]:
                    sample_node = self._datasets_bucket.get_sample(self._active_dataset.last_name, sample_id)
                    item_nodes = self._datasets_bucket.get_items(self._active_dataset.last_name, sample_id)
                    return sample_node, item_nodes

                # GRPC call
                EventLoop.defer_call(
                    _get_sample_with_data,
                    _deferred_get_sample,
                    fallback=self.show_error_dialog
                )

    def set_sample_image(self, data: bytes):
        """ Update preview image

        :param data: data bytes
        :type data: bytes
        """

        if data is None:  # Empyt bytes = No preview
            self.sample_image.setPixmap(QPixmap())
            return

        # Preview image
        self.pixmap = QPixmap()
        self.pixmap.loadFromData(data)
        size = self.sample_image.size()
        self.pixmap = self.pixmap.scaled(size.width(), size.height(), QtCore.Qt.KeepAspectRatio)
        # self.sample_image.setScaledContents(True)
        self.sample_image.setMinimumSize(1, 1)
        self.sample_image.setAlignment(QtCore.Qt.AlignCenter)
        self.sample_image.setPixmap(self.pixmap)

    def show_items_list(self):
        """ Shows active dataset/sample items list """

        # clear previous list
        self.items_list.clear()

        if self._active_sample is not None:
            for item_name in self._cached_items:
                self.items_list.addItem(QListWidgetItem(QIcon(get_absolute_path("data/icons/item.png")), f'{item_name}'))

        if self.items_list.count() > 0:
            self.items_list.setCurrentRow(0)

    def on_item_selected(self, index):
        """ On item selected event

        :param index: item index
        :type index: any
        """

        if self.items_list.currentItem() is not None:  # Selection is valid?
            if self._active_dataset is not None:  # Is dataset active?
                if self._active_sample is not None:  # Is Sample active?
                    item_name: str = self.items_list.currentItem().text()
                    if item_name in self._cached_items:
                        item: MNode = self._cached_items[item_name]
                        self.set_sample_image(item.get_data()[0])

    def new_dataset(self):
        """ Creates new dataset """

        # Dialog for dataset name
        name, status = QInputDialog.getText(self, "New Dataset", "Dataset name:")

        if status:  # Dialog OK

            # Creates dataset
            res = self._datasets_bucket.new_dataset(name)

            if res is not None:
                self.show_dataset_list()
            else:  # Error for dataset name collision
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Maybe name is used!')
                msg.setWindowTitle("Error creating dataset")
                msg.exec_()

    def on_dropped_folder(self, files: list):
        if len(files) != 1:
            pass
        else:
            folder = Path(files[0])
            if not folder.is_dir():
                show_error_message("Dropped item is not a folder!")
            else:
                files = sorted(list(folder.glob('*')))
                if self._active_dataset is not None:  # Spawns upload thread
                    self._upload_thread = UploadSamplesThread(self._datasets_bucket, self._active_dataset, folder)
                    self._upload_thread.PROGRESS_CHANGED.connect(self.on_upload_progress)
                    self._upload_thread.PROCESS_END.connect(self.on_upload_end)
                    self._upload_thread.start()

    def on_upload_progress(self, value: int):
        """ On upload progress event

        :param value: new progress value
        :type value: int
        """

        self.upload_progress.setValue(value)

    def on_upload_end(self, status: bool):
        """ On Upload end event

        :param status: TRUE for end
        :type status: bool
        """

        self.reload_dataset(self._active_dataset.last_name)


@click.command("Datasets Browser GUI")
@click.option("-c", "--database_configuration", default='database.yml', help="Database configuration file")
def datasets_browser(database_configuration):
    app = QApplication(sys.argv)
    cfg = yaml.safe_load(open(get_absolute_path(database_configuration), 'r'))
    client_cfg = MongoDatabaseClientCFG.from_dict(cfg)
    datasets_bucket = DatasetsBucket(client_cfg)
    w = DatasetsBrowser(datasets_bucket)
    w.show()
    w.setStyleSheet(open(get_absolute_path('data/datasets_browser.qss'), 'r').read())
    w.setWindowIcon(QIcon(get_absolute_path('data/icons/logo.png')))
    sys.exit(app.exec_())


if __name__ == '__main__':
    datasets_browser()
