import os
import random
from collections import namedtuple
from re import Pattern
from typing import Union, List

FileMatch = namedtuple('FileMatch', ['FilePath', 'Match'])


def find_all_files(base_dir: str,
                   patterns: Union[Pattern, List[Pattern]],
                   find_directories: bool = False) -> List[FileMatch]:
    matching_files = []
    patterns = patterns if isinstance(patterns, list) else [patterns]
    if find_directories:
        for root, dirs, files in os.walk(base_dir):
            for pattern in patterns:
                match = pattern.search(root)
                if match:
                    matching_files.append(FileMatch(root, match))
    else:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                for pattern in patterns:
                    match = pattern.search(file_path)
                    if match:
                        matching_files.append(FileMatch(file_path, match))
    return matching_files


def random_sample(lst: List, sample_size: int) -> List:
    if sample_size is None or sample_size >= len(lst):
        return lst
    sampled_indices = sorted(random.sample(range(len(lst)), sample_size))
    return [lst[i] for i in sampled_indices]
