import os
import random
from dataclasses import dataclass
from re import Pattern, Match
from typing import Union, List, TypeVar, Tuple

from natsort import natsorted

T = TypeVar('T')


@dataclass
class FileMatch:
    FilePath: str
    Match: Match[str]


def find_all_files(base_dir: str,
                   patterns: Union[Pattern, List[Pattern]],
                   find_directories: bool = False,
                   sort: bool = True) -> List[FileMatch]:
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
    if sort:
        # Sort the matching files by file path using natural sorting
        matching_files = natsorted(matching_files, key=lambda x: x.FilePath)
    return matching_files


def random_sample(lst: List[T], sample_size: int) -> List[T]:
    if sample_size is None or sample_size >= len(lst):
        return lst
    sampled_indices = sorted(random.sample(range(len(lst)), sample_size))
    return [lst[i] for i in sampled_indices]


def stratified_split(
    stratify: List[str], split_sample_sizes: Tuple[int, int, int]
) -> Tuple[List[int], List[int], List[int]]:
    total_samples = sum(split_sample_sizes)
    if len(stratify) != total_samples:
        raise ValueError(
            'Length of stratify list must match the total number of samples')

    # Determine unique classes
    unique_classes = set(stratify)
    class_counts = {label: 0 for label in unique_classes}
    for label in stratify:
        class_counts[label] += 1

    # Calculate the number of samples for each class in each split
    class_split_counts = {label: [0, 0, 0] for label in unique_classes}
    for label, count in class_counts.items():
        train_count = round(count * split_sample_sizes[0] / total_samples)
        val_count = round(count * split_sample_sizes[1] / total_samples)
        test_count = count - train_count - val_count
        class_split_counts[label] = [train_count, val_count, test_count]

    # Randomly sample indices for each class
    train_indices, val_indices, test_indices = [], [], []
    class_indices = {label: [] for label in unique_classes}
    for idx, label in enumerate(stratify):
        class_indices[label].append(idx)

    for label, indices in class_indices.items():
        random.shuffle(indices)
        train_count, val_count, test_count = class_split_counts[label]
        train_indices.extend(sorted(indices[:train_count]))
        val_indices.extend(sorted(indices[train_count:train_count +
                                          val_count]))
        test_indices.extend(sorted(indices[train_count + val_count:]))

    return train_indices, val_indices, test_indices


def insert_subdir(path: str, subdir: str) -> str:
    dir_name, file_name = os.path.split(path)
    new_path = os.path.join(dir_name, subdir, file_name)
    return new_path


def remove_subdir(path: str, subdir: str) -> str:
    parts = path.split(os.sep)
    if subdir in parts:
        parts.remove(subdir)
    new_path = os.sep.join(parts)
    return new_path
