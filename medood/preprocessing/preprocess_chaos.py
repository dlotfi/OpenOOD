import os
import re
from collections import namedtuple
from typing import List

from preprocessor_base import BaseDICOMPreProcessor, FilePair
from preprocessor_config import PreProcessorConfig
from utils import find_all_files, random_sample


class CHAOS_PreProcessor(BaseDICOMPreProcessor):
    def find_and_sample_dicom_series(self,
                                     split: str = None) -> List[FilePair]:
        pattern = re.compile(r'(Train_Sets|Test_Sets)/MR/(\d+)/'
                             r'T1DUAL/DICOM_anon/(InPhase|OutPhase)')

        candidate_series = find_all_files(self.cfg.base_dir,
                                          pattern,
                                          find_directories=True)
        self.logger.info(f'Found total {len(candidate_series)} DICOM series.')
        # Filter series based on the split type
        if split is not None:
            split_name = 'Train_Sets' if split == 'Train' else 'Test_Sets'
            candidate_series = [
                f for f in candidate_series if f.Match.group(1) == split_name
            ]
            self.logger.info(f'Total {len(candidate_series)} DICOM series'
                             f' are in the specified split: {split}.')
        candidate_series = random_sample(candidate_series,
                                         self.cfg.num_samples)

        # Pair each file with a target file name
        paired_files = []
        for dicom_series, match in candidate_series:
            split_part = 'Train' if match.group(1) == 'Train_Sets' else 'Test'
            number = match.group(2)
            phase = match.group(3)
            output_nifti_name = \
                f'CHAOS_{split_part}_{number}_{phase}_T1.nii.gz'
            output_nifti_path = os.path.join(self.cfg.output_dir,
                                             output_nifti_name)
            paired_files.append(FilePair(dicom_series, output_nifti_path))

        self.logger.info(f'Sampled {len(paired_files)} DICOM series.')
        return paired_files

    def run(self):
        self.logger.info('Start preprocessing CHAOS dataset')
        self.logger.info(self.cfg)
        sampled_dicom_series = self.find_and_sample_dicom_series()
        processed_files = self.convert_dicom_series_to_nifti(
            sampled_dicom_series)
        resampled_files = self.resample_and_center_crop_files(
            [f.Processed for f in processed_files])
        # todo: Questions:
        # 1. Normalize the images?
        FileProcessed = namedtuple('FileProcessed',
                                   ['Original', 'Processed', 'Resampled'])
        processed_files = [
            FileProcessed(o, p, (p in resampled_files))
            for o, p in processed_files
        ]
        self.log_processed_files(processed_files)
        self.logger.info('CHAOS dataset preprocessing completed.')


if __name__ == '__main__':
    cfg = PreProcessorConfig()
    cfg.parse_args()
    preprocessor = CHAOS_PreProcessor(cfg)
    preprocessor.run()
