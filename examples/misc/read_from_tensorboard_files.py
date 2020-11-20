from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.hparams import plugin_data_pb2

import numpy as np

from typing import Dict, Any
from pathlib import Path


def get_hparams_from_accumulator(event_accumulator: EventAccumulator) -> Dict[str, Any]:
    """Retrieves all hyperparameters from an event accumulator as a dictionary

    :param event_accumulator: event accumulator
    :type event_accumulator: EventAccumulator
    :return: dictionary mapping each hyperparameter name to its value
    :rtype: Dict[str, Any]
    """
    # Read plugin content
    hp_bytes = event_accumulator.PluginTagToContent('hparams')['_hparams_/session_start_info']

    # Build pb2 message from bytes
    hp_pb2 = plugin_data_pb2.HParamsPluginData.FromString(hp_bytes)

    # Message to python dict - is there an easier way?
    hp = {k: v.ListFields()[0][1] for k, v in dict(hp_pb2.ListFields()[0][1].hparams).items()}
    return hp


def get_hparams_from_multiplexer(event_multiplexer: EventMultiplexer) -> Dict[str, Dict[str, Any]]:
    """Retrieves all hyperparameters for every run of the given event multiplexer as a dictionary

    :param event_multiplexer: event multiplexer
    :type event_multiplexer: EventMultiplexer
    :return: dictionary mapping each run name to a dictionary containing all hyperparameters
    :rtype: Dict[str, Dict[str, Any]]
    """
    hparams = {}

    # Iterate on each run
    for run in event_multiplexer.Runs().keys():

        # Get event accumulator
        event_accumulator = event_multiplexer.GetAccumulator(run)

        # Get hparams from event accumulator
        hparams[run] = get_hparams_from_accumulator(event_accumulator)

    return hparams


def get_scalars_from_accumulator(event_accumulator: EventAccumulator) -> Dict[str, np.ndarray]:
    """Retrieves all scalars from an event accumulator as a dictionary

    :param event_accumulator: event accumulator
    :type event_accumulator: EventAccumulator
    :return: dictionary mapping each tag to a numpy array containing the scalar data
    :rtype: Dict[str, np.ndarray]
    """
    scalars = {}

    # Check if 'scalars' is a tag
    tags = event_accumulator.Tags()
    if 'scalars' in tags:

        # Iterate on each scalar tag
        for tag in tags['scalars']:

            # Get events from accumulator
            scalar_events = event_accumulator.Scalars(tag)

            # Convert events to numpy array
            scalars[tag] = np.array(scalar_events)

    return scalars


def get_scalars_from_multiplexer(event_multiplexer: EventMultiplexer) -> Dict[str, Dict[str, np.ndarray]]:
    """Retrieves all hyperparameters for every run of the given event multiplexer as a dictionary

    :param event_multiplexer: event multiplexer
    :type event_multiplexer: EventMultiplexer
    :return: dictionary mapping each run name to a dictionary containing all hyperparameters
    :rtype: Dict[str, Dict[str, Any]]
    """
    scalars = {}

    # Iterate on each run
    for run in event_multiplexer.Runs().keys():

        # Get event accumulator
        event_accumulator = event_multiplexer.GetAccumulator(run)

        # Get scalars from event accumulator
        scalars[run] = get_scalars_from_accumulator(event_accumulator)

    return scalars


def read_from_path(root_dir: Path):
    """Reads logged tensorboard data from a directory tree

    By default, tensorboard expects the result directory to have the following structure

    root_dir
    |- run_x
    |  |- events.out.tfevents.XXXXX (must contain 'tfevents' in its name)
    |  |- events.out.tfevents.XXXXX_2 (multiple tfevents files can be associated to the same run)
    |  |- ...
    |- run_y
    |  |- events.out.tfevents.YYYYY
    |  |- ...
    |  |- run_z
    |  |  |- events.out.tfevents.ZZZZZ (runs can be nested, nested name will be 'run_y/run_z')
    |  |  |- ...
    |  |- ...
    |- ...

    All events belonging to the same leaf directory will be merged into a single EventAccumulator.
    An EventMultiplexed contains multiple EventAccumulator, one for each leaf directory.

    :param root_dir: results root directory
    :type root_dir: Path
    """

    # Instantiate EventMultiplexer
    event_multiplexer = EventMultiplexer()

    # Fetch runs from directory tree
    event_multiplexer = event_multiplexer.AddRunsFromDirectory(str(root_dir))

    # Load data
    event_multiplexer.Reload()

    # Find hparams for each run
    hparams = get_hparams_from_multiplexer(event_multiplexer)
    scalars = get_scalars_from_multiplexer(event_multiplexer)
    return hparams, scalars


if __name__ == "__main__":
    path = Path('/home/luca/Desktop/test_tensorboard')
    hparams = read_from_path(path)
    print(hparams)
