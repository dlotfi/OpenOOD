import os
import re
from collections import namedtuple
from typing import List

from preprocessor_base import BaseDICOMPreProcessor, FilePair
from preprocessor_config import PreProcessorConfig
from utils import find_all_files, random_sample


class LUMBAR_PreProcessor(BaseDICOMPreProcessor):
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
        for dicom_series, match in candidate_series:
            number = match.group(1)
            output_nifti_name = f'LUMBAR_{number}_T1.nii.gz'
            output_nifti_path = os.path.join(self.cfg.output_dir,
                                             output_nifti_name)
            paired_files.append(FilePair(dicom_series, output_nifti_path))

        self.logger.info(f'Sampled {len(paired_files)} DICOM series.')
        return paired_files

    def run(self):
        self.logger.info('Start preprocessing LUMBAR dataset')
        self.logger.info(self.cfg)
        sampled_dicom_series = self.find_and_sample_dicom_series()
        processed_files = self.convert_dicom_series_to_nifti(
            sampled_dicom_series)
        resampled_files = self.resample_and_center_crop_files(
            [f.Processed for f in processed_files])
        # todo: Questions:
        # 1. Normalize the images?
        # 2. Which folder? T1_TSE_TRA_000*?
        FileProcessed = namedtuple('FileProcessed',
                                   ['Original', 'Processed', 'Resampled'])
        processed_files = [
            FileProcessed(o, p, (p in resampled_files))
            for o, p in processed_files
        ]
        self.log_processed_files(processed_files)
        self.logger.info('LUMBAR dataset preprocessing completed.')


if __name__ == '__main__':
    cfg = PreProcessorConfig()
    cfg.parse_args()
    preprocessor = LUMBAR_PreProcessor(cfg)
    preprocessor.run()
