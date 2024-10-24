import os
import re
from typing import List

from preprocessor_base import BaseBrainPreProcessor, FilePair
from preprocessor_config import PreProcessorBrainConfig
from utils import find_all_files, random_sample


class LUMIERE_PreProcessor(BaseBrainPreProcessor):
    def find_and_sample_files(self,
                              splits: List[str] = None) -> List[FilePair]:
        pattern = re.compile(
            r'Patient-(\d{3})/(week-\d{3}(-\d+)?)/T1\.nii\.gz')

        candidate_files = find_all_files(self.cfg.base_dir, pattern)
        self.logger.info(f'Found total {len(candidate_files)} files.')
        # Filter files based on the split
        if splits is not None:
            candidate_files = [
                f for f in candidate_files if f.Match.group(2) in splits
            ]
            self.logger.info(f'Total {len(candidate_files)} files'
                             f' are in the specified splits: {splits}.')
        candidate_files = random_sample(candidate_files, self.cfg.num_samples)

        # Pair each file with a target file name
        paired_files = []
        for file in candidate_files:
            number = file.Match.group(1)
            split_part = file.Match.group(2)
            output_name = f'LUMIERE_{number}_{split_part}_T1.nii.gz'
            output_path = os.path.join(self.cfg.output_dir, output_name)
            paired_files.append(FilePair(file.FilePath, output_path))

        self.logger.info(f'Sampled {len(paired_files)} files.')
        return paired_files

    def run(self):
        self.logger.info('Start preprocessing LUMIERE dataset')
        self.logger.info(self.cfg)
        # 1. Find all files from 'week-000' or 'week-000-1' folders
        #    and sample randomly from them
        sampled_files = self.find_and_sample_files(
            splits=['week-000', 'week-000-1'])
        # 2. Register to SRI24, and normalize all sampled images
        processed_files = self.process_brain_images(sampled_files,
                                                    skull_stripping=False)
        self.save_processed_files(processed_files)
        self.logger.info('LUMIERE dataset preprocessing completed.')
        # Questions:
        # * Which folder: 'week-000' and 'week-000-1'?


if __name__ == '__main__':
    cfg = PreProcessorBrainConfig()
    cfg.parse_args()
    preprocessor = LUMIERE_PreProcessor(cfg)
    preprocessor.run()
