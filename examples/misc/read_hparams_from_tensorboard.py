from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.hparams import plugin_data_pb2

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


def get_hparams_from_multiplexer(em: EventMultiplexer) -> Dict[str, Dict[str, Any]]:
    """Retrieves all hyperparameters for every run of the given event multiplexer as a dictionary

    :param em: event multiplexer
    :type em: EventMultiplexer
    :return: dictionary mapping each run name to a dictionary containing all hyperparameters
    :rtype: Dict[str, Dict[str, Any]]
    """
    hparams = {}

    for run in em.Runs().keys():

        # Get event accumulator for each run
        ea = em.GetAccumulator(run)

        # Get hparams from event accumulator
        hparams[run] = get_hparams_from_accumulator(ea)

    return hparams


def hparams_from_path(root_dir: Path):
    """Reads logged tensorboard hyperparameters from a directory tree

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
    em = EventMultiplexer()

    # Fetch runs from directory tree
    em = em.AddRunsFromDirectory(str(root_dir))

    # Load data
    em.Reload()

    # Find hparams for each run
    hparams = get_hparams_from_multiplexer(em)
    return hparams


if __name__ == "__main__":
    path = Path('/home/luca/Desktop/test_tensorboard')
    hparams = hparams_from_path(path)
    print(hparams)
