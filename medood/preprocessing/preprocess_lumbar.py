import os
import re
from dataclasses import dataclass
from typing import List

from preprocessor_base import (BaseDICOMPreProcessor, BaseNonBrainPreProcessor,
                               FilePair)
from preprocessor_config import PreProcessorConfig
from utils import find_all_files, random_sample


@dataclass
class ResampledFilePair(FilePair):
    Resampled: bool


class LUMBAR_PreProcessor(BaseDICOMPreProcessor, BaseNonBrainPreProcessor):
    def find_and_sample_dicom_series(self) -> List[FilePair]:
        pattern = re.compile(r'(\d{4})/.*/T1_TSE_TRA_000\d')

        candidate_series = find_all_files(self.cfg.base_dir,
                                          pattern,
                                          find_directories=True)
        self.logger.info(f'Found total {len(candidate_series)} DICOM series.')
        candidate_series = random_sample(candidate_series,
                                         self.cfg.num_samples)

        # Pair each file with a target file name
        paired_files = []
        for series in candidate_series:
            number = series.Match.group(1)
            output_nifti_name = f'LUMBAR_{number}_T1.nii.gz'
            output_nifti_path = os.path.join(self.cfg.output_dir,
                                             output_nifti_name)
            paired_files.append(FilePair(series.FilePath, output_nifti_path))

        self.logger.info(f'Sampled {len(paired_files)} DICOM series.')
        return paired_files

    def run(self):
        self.logger.info('Start preprocessing LUMBAR dataset')
        self.logger.info(self.cfg)
        # 1. Find all DICOM series and sample randomly from them
        sampled_dicom_series = self.find_and_sample_dicom_series()
        # 2. Convert all sampled series to NIfTI while normalizing them
        processed_files = self.convert_dicom_series_to_nifti(
            sampled_dicom_series, normalize=True)
        # 3. Resample all volumes to 1mm x 1mm x 1mm isotropic and
        #    center crop a 240 x 240 x 155 volume
        resampled_files = self.resample_and_center_crop_files(
            [f.Output for f in processed_files])
        processed_files = [
            ResampledFilePair(f.Source, f.Output, f.Output in resampled_files)
            for f in processed_files
        ]
        self.save_processed_files(processed_files)
        self.logger.info('LUMBAR dataset preprocessing completed.')
        # Questions:
        # *  Which folder: 'T1_TSE_TRA_000*'?


if __name__ == '__main__':
    cfg = PreProcessorConfig()
    cfg.parse_args()
    preprocessor = LUMBAR_PreProcessor(cfg)
    preprocessor.run()
