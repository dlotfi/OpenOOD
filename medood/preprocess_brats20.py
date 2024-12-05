import os
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd

from preprocessor_base import BaseBrainPreProcessor, FilePair
from preprocessor_config import PreProcessorBrainConfig
from utils import stratified_split


@dataclass
class SplitLabeledFilePair(FilePair):
    Split: str
    Label: str


class BraTS20_PreProcessor(BaseBrainPreProcessor):
    def __init__(self, cfg: PreProcessorBrainConfig, output_dirs: Dict[str,
                                                                       str]):
        super().__init__(cfg)
        self.output_dirs = output_dirs

    def find_and_sample_files(self) -> Dict[str, List[SplitLabeledFilePair]]:
        csv_path = os.path.join(self.cfg.base_dir, 'name_mapping.csv')
        df = pd.read_csv(csv_path)

        labels = []
        subject_numbers = []
        for _, row in df.iterrows():
            subject_numbers.append(row['BraTS_2020_subject_ID'].split('_')[-1])
            labels.append(row['Grade'])

        train_indices, val_indices, test_indices = \
            stratified_split(labels, self.cfg.split_num_samples)

        split_indices = {
            'TRAIN': train_indices,
            'VALIDATION': val_indices,
            'TEST': test_indices
        }

        # Pair each file with a target file name
        paired_files = {k: [] for k in self.output_dirs.keys()}
        file_suffixes = {'t1': 't1', 't2': 't2', 't1c': 't1ce', 't2f': 'flair'}
        for split, indices in split_indices.items():
            for idx in indices:
                subject_number = subject_numbers[idx]
                label = labels[idx]
                for key in paired_files.keys():
                    if key != 't1' and split == 'TRAIN':
                        continue
                    input_path = os.path.join(
                        self.cfg.base_dir,
                        f'BraTS20_Training_{subject_number}',
                        f'BraTS20_Training_{subject_number}_'
                        f'{file_suffixes[key]}.nii.gz')
                    output_path = os.path.join(
                        self.output_dirs[key],
                        f'BraTS20_{subject_number}_{key.upper()}.nii.gz')
                    paired_files[key].append(
                        SplitLabeledFilePair(input_path, output_path, split,
                                             label))

        for key, files in paired_files.items():
            self.logger.info(f'Sampled {len(files)} {key.upper()} files (' + (
                f'TRAIN: {len(train_indices)}, ' if key == 't1' else '') +
                             f'VALIDATION: {len(val_indices)}, ' +
                             f'TEST: {len(test_indices)}).')
        return paired_files

    def run(self):
        self.logger.info('Start preprocessing BraTS2020 dataset')
        self.logger.info(self.cfg)
        # 1. Find all files in 'Train' split and randomly split
        #    them into Train, Validation, and Test
        sampled_files = self.find_and_sample_files()
        # 2. Normalize all T1, T2, T1C, and T2-FLAIR sampled images
        for key in sampled_files.keys():
            os.makedirs(self.output_dirs[key], exist_ok=True)
            processed_files = self.normalize_images(sampled_files[key])
            csv_path = os.path.join(self.output_dirs[key],
                                    'processed_files.csv')
            self.save_processed_files(processed_files, csv_path)
        self.logger.info('BraTS2020 dataset preprocessing completed.')


if __name__ == '__main__':
    cfg = PreProcessorBrainConfig()
    cfg.parser.add_argument(
        '--split_num_samples',
        type=int,
        nargs=3,
        required=True,
        help='Train, Validation, Test splits number of samples')
    cfg.parser.add_argument(
        '--extra_output_dirs',
        type=str,
        nargs='+',
        required=True,
        help='Output directories for the processed data in '
        'modality_name=path format (e.g., t2=path/to/t2 '
        't1c=path/to/t1c t2f=path/to/t2f)')
    cfg.parse_args()
    output_dirs = {'t1': cfg.output_dir}
    output_dirs.update(
        {kv.split('=')[0]: kv.split('=')[1]
         for kv in cfg.extra_output_dirs})
    if cfg.num_samples is not None:
        if sum(cfg.split_num_samples) != cfg.num_samples:
            raise ValueError(
                f'The sum of split_num_samples {cfg.split_num_samples}'
                f' must be equal to num_samples {cfg.num_samples}.')
    preprocessor = BraTS20_PreProcessor(cfg, output_dirs)
    preprocessor.run()
