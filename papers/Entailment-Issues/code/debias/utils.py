'''
The code is adapted from https://github.com/UKPLab/emnlp2020-debiasing-unknown/blob/main/src/utils.py

License: Apache License 2.0
'''

import logging
import pickle
import sys
from os import makedirs
from os.path import dirname
from typing import TypeVar

from multiprocessing import Lock
from multiprocessing import Pool
from typing import Iterable, List

import requests

from tqdm import tqdm


T = TypeVar('T')


def add_stdout_logger():
    """Setup stdout logging"""

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S', )
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def ensure_dir_exists(filename):
    """Make sure the parent directory of `filename` exists"""
    makedirs(dirname(filename), exist_ok=True)


def download_to_file(url, output_file):
    """Download `url` to `output_file`, intended for small files."""
    ensure_dir_exists(output_file)
    with requests.get(url) as r:
        r.raise_for_status()
        with open(output_file, 'wb') as f:
            f.write(r.content)


def load_pickle(filename):
    """Load an object from a pickled file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


# ++++++++++++++++++++++++++++++++++++++++++++++++++
# data processing functions

def flatten_list(iterable_of_lists: Iterable[Iterable[T]]) -> List[T]:
    """Unpack lists into a single list."""
    return [x for sublist in iterable_of_lists for x in sublist]


def split(lst: List[T], n_groups) -> List[List[T]]:
    """ partition `lst` into `n_groups` that are as evenly sized as possible  """
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups


def group(lst: List[T], max_group_size) -> List[List[T]]:
    """partition `lst` into that the mininal number of groups that as evenly sized
    as possible  and are at most `max_group_size` in size """
    if max_group_size is None:
        return [lst]
    n_groups = (len(lst) + max_group_size - 1) // max_group_size
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups


class Processor:

    def process(self, data: Iterable):
        """Map elements to an unspecified output type, the output but type must None or
        be able to be aggregated with the  `+` operator"""
        raise NotImplementedError()

    def finalize_chunk(self, data):
        """Finalize the output from `preprocess`, in multi-processing senarios this will still be run on
         the main thread so it can be used for things like interning"""
        pass


def _process_and_count(questions: List, preprocessor: Processor):
    count = len(questions)
    output = preprocessor.process(questions)
    return output, count


def process_par(data: List, processor: Processor, n_processes,
                chunk_size=1000, desc=None, initializer=None):
    """Runs `processor` on the elements in `data`, possibly in parallel, and monitor with tqdm"""

    if chunk_size <= 0:
        raise ValueError("Chunk size must be >= 0, but got %s" % chunk_size)
    if n_processes is not None and n_processes <= 0:
        raise ValueError("n_processes must be >= 1 or None, but got %s" % n_processes)
    n_processes = min(len(data), 1 if n_processes is None else n_processes)

    if n_processes == 1 and not initializer:
        out = processor.process(tqdm(data, desc=desc, ncols=80))
        processor.finalize_chunk(out)
        return out
    else:
        chunks = split(data, n_processes)
        chunks = flatten_list([group(c, chunk_size) for c in chunks])
        total = len(data)
        pbar = tqdm(total=total, desc=desc, ncols=80)
        lock = Lock()

        def call_back(results):
            processor.finalize_chunk(results[0])
            with lock:
                pbar.update(results[1])

        with Pool(n_processes, initializer=initializer) as pool:
            results = [
                pool.apply_async(_process_and_count, [c, processor], callback=call_back)
                for c in chunks
            ]
            results = [r.get()[0] for r in results]

        pbar.close()
        output = results[0]
        if output is not None:
            for r in results[1:]:
                output += r
        return output
