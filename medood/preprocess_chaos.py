import os
import re
from typing import List

from preprocessor_base import (BaseDICOMPreProcessor, BaseNonBrainPreProcessor,
                               FilePair, TempFilePair)
from preprocessor_config import PreProcessorConfig
from utils import (find_all_files, random_sample, insert_subdir, remove_subdir)


class CHAOS_PreProcessor(BaseDICOMPreProcessor, BaseNonBrainPreProcessor):
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
        for series in candidate_series:
            split_part = \
                'Train' if series.Match.group(1) == 'Train_Sets' else 'Test'
            number = series.Match.group(2)
            phase = series.Match.group(3)
            output_nifti_name = \
                f'CHAOS_{split_part}_{number}_{phase}_T1.nii.gz'
            output_nifti_path = os.path.join(self.cfg.output_dir,
                                             output_nifti_name)
            paired_files.append(FilePair(series.FilePath, output_nifti_path))

        self.logger.info(f'Sampled {len(paired_files)} DICOM series.')
        return paired_files

    def run(self):
        self.logger.info('Start preprocessing CHAOS dataset')
        self.logger.info(self.cfg)
        subdir = 'raw_nifti'
        # 1. Find all DICOM series in both 'Train' and 'Test' splits
        #    and sample randomly from them
        sampled_dicom_series = self.find_and_sample_dicom_series()
        # 2. Convert all sampled series to NIfTI while normalizing them
        os.makedirs(os.path.join(self.cfg.output_dir, subdir), exist_ok=True)
        sampled_dicom_series = [
            FilePair(f.Source, insert_subdir(f.Output, subdir))
            for f in sampled_dicom_series
        ]
        converted_files = self.convert_dicom_series_to_nifti(
            sampled_dicom_series)
        # 3. Resample all volumes to 1mm isotropic and center crop
        files_to_be_processed = [
            TempFilePair(Source=f.Output,
                         Output=remove_subdir(f.Output, subdir),
                         OriginalSource=f.Source) for f in converted_files
        ]
        processed_files = self.resample_and_center_crop_images(
            files_to_be_processed, normalize=True)
        self.save_processed_files(
            [FilePair(f.OriginalSource, f.Output) for f in processed_files])
        self.logger.info('CHAOS dataset preprocessing completed.')
        # Questions:
        # * Which split: Train, Test or both?


if __name__ == '__main__':
    cfg = PreProcessorConfig()
    cfg.parse_args()
    preprocessor = CHAOS_PreProcessor(cfg)
    preprocessor.run()
