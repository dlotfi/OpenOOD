import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from preprocessor_base import BaseBrainPreProcessor, FilePair
from preprocessor_config import PreProcessorBrainConfig
from utils import stratified_split


@dataclass
class SplitLabeledFilePair(FilePair):
    Split: str
    Label: str


class BraTS20_PreProcessor(BaseBrainPreProcessor):
    def find_and_sample_files(self) -> \
            Tuple[List[SplitLabeledFilePair], List[FilePair], List[FilePair]]:
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
        t1_paired_files = []
        t1c_paired_files = []
        t2f_paired_files = []
        for split, indices in split_indices.items():
            for idx in indices:
                subject_number = subject_numbers[idx]
                label = labels[idx]
                # T1 source and output pair
                input_path = os.path.join(
                    self.cfg.base_dir, f'BraTS20_Training_{subject_number}',
                    f'BraTS20_Training_{subject_number}_t1.nii.gz')
                output_path = os.path.join(
                    self.cfg.output_dir, f'BraTS20_{subject_number}_T1.nii.gz')
                t1_paired_files.append(
                    SplitLabeledFilePair(input_path, output_path, split,
                                         label))
                if split == 'TRAIN':
                    continue
                # T1C source and output pair
                t1c_input_path = input_path.replace('t1', 't1ce')
                t1c_output_path = os.path.join(
                    self.cfg.output_dir_t1c,
                    f'BraTS20_{subject_number}_T1C.nii.gz')
                t1c_paired_files.append(
                    SplitLabeledFilePair(t1c_input_path, t1c_output_path,
                                         split, label))
                # T2-FLAIR source and output pair
                t2f_input_path = input_path.replace('t1', 'flair')
                t2f_output_path = os.path.join(
                    self.cfg.output_dir_t2f,
                    f'BraTS20_{subject_number}_T2F.nii.gz')
                t2f_paired_files.append(
                    SplitLabeledFilePair(t2f_input_path, t2f_output_path,
                                         split, label))

        self.logger.info(f'Sampled {len(t1_paired_files)} T1 files '
                         f'(TRAIN: {len(train_indices)}, '
                         f'VALIDATION: {len(val_indices)}, '
                         f'TEST: {len(test_indices)}).')
        self.logger.info(f'Sampled {len(t1c_paired_files)} T1C files.'
                         f'(VALIDATION: {len(val_indices)}, '
                         f'TEST: {len(test_indices)}).')
        self.logger.info(f'Sampled {len(t2f_paired_files)} T2-FLAIR files.'
                         f'(VALIDATION: {len(val_indices)}, '
                         f'TEST: {len(test_indices)}).')
        return t1_paired_files, t1c_paired_files, t2f_paired_files

    def run(self):
        self.logger.info('Start preprocessing BraTS2020 dataset')
        self.logger.info(self.cfg)
        # 1. Find all files in 'Train' split and randomly split
        #    them into Train, Validation, and Test
        t1_sampled_files, t1c_sampled_files, t2f_sampled_files = \
            self.find_and_sample_files()
        # 2. Normalize all T1 sampled images
        t1_processed_files = self.normalize_images(t1_sampled_files)
        self.save_processed_files(t1_processed_files)
        #    Normalize all T1C (post contrast T1) sampled images
        t1c_processed_files = self.normalize_images(t1c_sampled_files)
        csv_path = os.path.join(self.cfg.output_dir_t1c, 'processed_files.csv')
        self.save_processed_files(t1c_processed_files, csv_path)
        #    Normalize all T2-FLAIR sampled images
        t2f_processed_files = self.normalize_images(t2f_sampled_files)
        csv_path = os.path.join(self.cfg.output_dir_t2f, 'processed_files.csv')
        self.save_processed_files(t2f_processed_files, csv_path)
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
        '--output_dir_t1c',
        type=str,
        required=True,
        help='Output directory for the processed post contrast T1 data')
    cfg.parser.add_argument(
        '--output_dir_t2f',
        type=str,
        required=True,
        help='Output directory for the processed T2-FLAIR data')
    cfg.parse_args()
    if cfg.num_samples is not None:
        if sum(cfg.split_num_samples) != cfg.num_samples:
            raise ValueError(
                f'The sum of split_num_samples {cfg.split_num_samples}'
                f' must be equal to num_samples {cfg.num_samples}.')
    preprocessor = BraTS20_PreProcessor(cfg)
    preprocessor.run()
