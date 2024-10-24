import os
import re
from typing import List

from preprocessor_base import BaseBrainPreProcessor, FilePair
from preprocessor_config import PreProcessorBrainConfig
from utils import find_all_files, random_sample


class EPISURG_PreProcessor(BaseBrainPreProcessor):
    def find_and_sample_files(self, split: str = None) -> List[FilePair]:
        pattern = re.compile(
            r'(sub-\d{4})/(preop|postop)/(\1_\2)-t1mri-\d\.nii\.gz')

        candidate_files = find_all_files(self.cfg.base_dir, pattern)
        self.logger.info(f'Found total {len(candidate_files)} files.')
        # Filter files based on the split type
        if split is not None:
            candidate_files = [
                f for f in candidate_files if f.Match.group(2) == split
            ]
            self.logger.info(f'Total {len(candidate_files)} files'
                             f' are in the specified split: {split}.')
        candidate_files = random_sample(candidate_files, self.cfg.num_samples)

        # Pair each file with a target file name
        paired_files = []
        for file in candidate_files:
            sub_and_split_part = file.Match.group(3)
            output_name = f'EPISURG_{sub_and_split_part}_T1.nii.gz'
            output_path = os.path.join(self.cfg.output_dir, output_name)
            paired_files.append(FilePair(file.FilePath, output_path))

        self.logger.info(f'Sampled {len(paired_files)} files.')
        return paired_files

    def run(self):
        self.logger.info('Start preprocessing EPISURG dataset')
        self.logger.info(self.cfg)
        # 1. Find all files from 'postop' folders and sample randomly from them
        #    (postop patients have a cavity in place of the epilepsy area)
        sampled_files = self.find_and_sample_files(split='postop')
        # 2. Register to SRI24, skull-strip, and normalize all sampled images
        processed_files = self.register_skullstrip_normalize_images(
            sampled_files)
        self.save_processed_files(processed_files)
        self.logger.info('EPISURG dataset preprocessing completed.')
        # Questions:
        # * Which folder: 'preop', 'postop' or both?


if __name__ == '__main__':
    cfg = PreProcessorBrainConfig()
    cfg.parse_args()
    preprocessor = EPISURG_PreProcessor(cfg)
    preprocessor.run()
